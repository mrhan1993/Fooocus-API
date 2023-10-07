from typing import Annotated, List
from fastapi import Depends, FastAPI, Header, Query, Response, UploadFile
import uvicorn
from fooocusapi.api_utils import narray_to_base64img, narray_to_bytesimg
from fooocusapi.models import GeneratedImageBase64, ImgUpscaleOrVaryRequest, Text2ImgRequest
from fooocusapi.task_queue import TaskQueue
from fooocusapi.worker import process_generate

app = FastAPI()

task_queue = TaskQueue()


@app.post("/v1/generation/text-to-image", response_model=List[GeneratedImageBase64], responses={
    200: {
        "description": "PNG bytes if request's 'Accept' header is 'image/png', otherwise JSON",
        "content": {
            "application/json": {
                "example": [{
                    "base64": "...very long string...",
                    "seed": 1050625087,
                    "finish_reason": "SUCCESS"
                }]
            },
            "image/png": {
                "example": "PNG bytes, what did you expect?"
            }
        }
    }
})
def text2img_generation(req: Text2ImgRequest, accept: str = Header(None),
                        accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if accept == 'image/png':
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    results = process_generate(req)

    if streaming_output:
        bytes = narray_to_bytesimg(results[0].im)
        return Response(bytes, media_type='image/png')
    else:
        results = [GeneratedImageBase64(base64=narray_to_base64img(
            item.im), seed=item.seed, finish_reason=item.finish_reason) for item in results]
        return results


@app.post("/v1/generation/image-uov", response_model=List[GeneratedImageBase64], responses={
    200: {
        "description": "PNG bytes if request's 'Accept' header is 'image/png', otherwise JSON",
        "content": {
            "application/json": {
                "example": [{
                    "base64": "...very long string...",
                    "seed": 1050625087,
                    "finish_reason": "SUCCESS"
                }]
            },
            "image/png": {
                "example": "PNG bytes, what did you expect?"
            }
        }
    }
})
def img_upscale_or_vary(input_image: UploadFile, req: ImgUpscaleOrVaryRequest = Depends(ImgUpscaleOrVaryRequest.as_form),
                        accept: str = Header(None),
                        accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if accept == 'image/png':
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    results = process_generate(req)

    if streaming_output:
        bytes = narray_to_bytesimg(results[0].im)
        return Response(bytes, media_type='image/png')
    else:
        results = [GeneratedImageBase64(base64=narray_to_base64img(
            item.im), seed=item.seed, finish_reason=item.finish_reason) for item in results]
        return results


def start_app(args):
    uvicorn.run("fooocusapi.api:app", host=args.host,
                port=args.port, log_level=args.log_level)
