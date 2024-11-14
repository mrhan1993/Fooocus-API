"""
Entry for startup fastapi server
"""
import logging
from fastapi import FastAPI, Header, Response, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

import uvicorn

from fooocusapi.utils import file_utils
from fooocusapi.routes.generate_v1 import secure_router as generate_v1
from fooocusapi.routes.generate_v2 import secure_router as generate_v2
from fooocusapi.routes.query import secure_router as query
from fooocusapi.utils.img_utils import convert_image


app = FastAPI()

logging.basicConfig(level=logging.DEBUG)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 记录错误信息到日志
    logging.error(f"Validation error: {exc.errors()}, body: {exc.body}")
    
    # 返回更详细的错误信息给客户端
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

app.add_middleware(GZipMiddleware, minimum_size=1000)
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
