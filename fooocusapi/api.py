"""
Entry for startup fastapi server
"""
from fastapi import FastAPI, Header, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from fooocusapi.utils import file_utils
from fooocusapi.routes.generate_v1 import secure_router as generate_v1
from fooocusapi.routes.generate_v2 import secure_router as generate_v2
from fooocusapi.routes.query import secure_router as query
from fooocusapi.utils.img_utils import convert_image


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from all sources
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all request headers
)


@app.get("/files/{date}/{file_name}", tags=["Query"])
async def get_output(date: str, file_name: str, accept: str = Header(None)):
    """
    Get a specific output by its ID.
    """
    accept_formats = ('png', 'jpg', 'jpeg', 'webp')
    try:
        _, ext = accept.lower().split("/")
        if ext not in accept_formats:
            ext = None
    except ValueError:
        ext = None

    if not file_name.endswith(accept_formats):
        return Response(status_code=404)

    if ext is None:
        try:
            return FileResponse(f"{file_utils.output_dir}/{date}/{file_name}")
        except FileNotFoundError:
            return Response(status_code=404)
    img = await convert_image(f"{file_utils.output_dir}/{date}/{file_name}", ext)
    return Response(content=img, media_type=f"image/{ext}")


app.include_router(query)
app.include_router(generate_v1)
app.include_router(generate_v2)


def start_app(args):
    """Start the FastAPI application"""
    file_utils.STATIC_SERVER_BASE = args.base_url + "/files/"
    uvicorn.run(
        app="fooocusapi.api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level)
