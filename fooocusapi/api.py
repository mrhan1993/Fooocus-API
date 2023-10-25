from typing import List, Optional
from fastapi import Depends, FastAPI, Header, Query, Response, UploadFile
from fastapi.params import File
import uvicorn
from fooocusapi.api_utils import generation_output, req_to_params
from fooocusapi.models import AsyncJobResponse, GeneratedImageBase64, ImgInpaintOrOutpaintRequest, ImgPromptRequest, ImgUpscaleOrVaryRequest, Text2ImgRequest
from fooocusapi.parameters import GenerationFinishReason, ImageGenerationResult
from fooocusapi.task_queue import TaskType
from fooocusapi.worker import process_generate, task_queue
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

work_executor = ThreadPoolExecutor(max_workers=task_queue.queue_size*2, thread_name_prefix="worker_")

img_generate_responses = {
    "200": {
        "description": "PNG bytes if request's 'Accept' header is 'image/png', otherwise JSON",
        "content": {
            "application/json": {
                "example": [{
                    "base64": "...very long string...",
                    "seed": 1050625087,
                    "finish_reason": "SUCCESS"
                }]
            },
            "application/json async": {
                "example": {
                    "job_id": 1,
                    "job_type": "Text to Image"
                }
            },
            "image/png": {
                "example": "PNG bytes, what did you expect?"
            }
        }
    }
}

def call_worker(req: Text2ImgRequest, accept: str):
    task_type = TaskType.text_2_img
    if isinstance(req, ImgUpscaleOrVaryRequest):
        task_type = TaskType.img_uov
    elif isinstance(req, ImgInpaintOrOutpaintRequest):
        task_type = TaskType.img_inpaint_outpaint
    elif isinstance(req, ImgPromptRequest):
        task_type = TaskType.img_prompt

    params = req_to_params(req)
    queue_task = task_queue.add_task(task_type, {'params': params.__dict__, 'accept': accept})

    if queue_task is None:
        print("[Task Queue] The task queue has reached limit")
        results = [ImageGenerationResult(im=None, seed=0,
                           finish_reason=GenerationFinishReason.queue_is_full)]
    elif req.async_process:
        work_executor.submit(process_generate, queue_task, params)
        results = queue_task
    else:
        results = process_generate(queue_task, params)

    return results

@app.get("/")
def home():
    return Response(content='Swagger-UI to: <a href="/docs">/docs</a>', media_type="text/html")


@app.post("/v1/generation/text-to-image", response_model=List[GeneratedImageBase64] | AsyncJobResponse, responses=img_generate_responses)
def text2img_generation(req: Text2ImgRequest, accept: str = Header(None),
                        accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if accept == 'image/png':
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    results = call_worker(req, accept)
    return generation_output(results, streaming_output)


@app.post("/v1/generation/image-upscale-vary", response_model=List[GeneratedImageBase64] | AsyncJobResponse, responses=img_generate_responses)
def img_upscale_or_vary(input_image: UploadFile, req: ImgUpscaleOrVaryRequest = Depends(ImgUpscaleOrVaryRequest.as_form),
                        accept: str = Header(None),
                        accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if accept == 'image/png':
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    results = call_worker(req, accept)
    return generation_output(results, streaming_output)


@app.post("/v1/generation/image-inpait-outpaint", response_model=List[GeneratedImageBase64] | AsyncJobResponse, responses=img_generate_responses)
def img_inpaint_or_outpaint(input_image: UploadFile, req: ImgInpaintOrOutpaintRequest = Depends(ImgInpaintOrOutpaintRequest.as_form),
                            accept: str = Header(None),
                            accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if accept == 'image/png':
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    results = call_worker(req, accept)
    return generation_output(results, streaming_output)


@app.post("/v1/generation/image-prompt", response_model=List[GeneratedImageBase64] | AsyncJobResponse, responses=img_generate_responses)
def img_prompt(cn_img1: Optional[UploadFile] = File(None),
               req: ImgPromptRequest = Depends(ImgPromptRequest.as_form),
               accept: str = Header(None),
               accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if accept == 'image/png':
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    results = call_worker(req, accept)
    return generation_output(results, streaming_output)


@app.get("/v1/generation/query-job", response_model=AsyncJobResponse, description="Query async generation job")
def query_job(job_id: int):
    queue_task = task_queue.get_task(job_id, True)
    if queue_task is None:
        return Response(content="Job not found", status_code=404)
    
    return generation_output(queue_task, False)


def start_app(args):
    uvicorn.run("fooocusapi.api:app", host=args.host,
                port=args.port, log_level=args.log_level)
