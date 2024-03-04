"""Fastapi routes for API"""

import uvicorn

from fastapi import Depends, FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fooocusapi.models.common.response import StopResponse

from fooocusapi.utils.api_utils import (
    api_key_auth,
    stop_worker,
)

from fooocusapi.utils import file_utils

from fooocusapi.routes.generate_v1 import secure_router as generate_v1
from fooocusapi.routes.generate_v2 import secure_router as generate_v2
from fooocusapi.routes.query import secure_router as query


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from all sources
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all request headers
)

secure_router = APIRouter(dependencies=[Depends(api_key_auth)])

@secure_router.post("/v1/generation/stop",
                    response_model=StopResponse,
                    description="Job stoping")
def stop():
    """Job stoping"""
    stop_worker()
    return StopResponse(msg="success")


app.mount("/files", StaticFiles(directory=file_utils.output_dir), name="files")

app.include_router(secure_router)
app.include_router(generate_v1)
app.include_router(generate_v2)
app.include_router(query)


def start_app(args):
    """Start the app"""
    file_utils.static_serve_base_url = args.base_url + "/files/"
    uvicorn.run(
        "fooocusapi.api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level
    )
