import uvicorn

from typing import List, Optional
from fastapi import Depends, FastAPI, Header, Query, Response, UploadFile, APIRouter, Depends
from fastapi.params import File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from fooocusapi.args import args
from fooocusapi.models import *
from fooocusapi.api_utils import req_to_params, generate_async_output, generate_streaming_output, generate_image_result_output, api_key_auth
import fooocusapi.file_utils as file_utils
from fooocusapi.parameters import GenerationFinishReason, ImageGenerationResult
from fooocusapi.task_queue import TaskType
from fooocusapi.worker import worker_queue, process_top, blocking_get_task_result
from fooocusapi.models_v2 import *
from fooocusapi.img_utils import base64_to_stream, read_input_image

from modules.util import HWC3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from all sources
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all request headers
)

secure_router = APIRouter(
    dependencies=[Depends(api_key_auth)]
)

img_generate_responses = {
    "200": {
        "description": "PNG bytes if request's 'Accept' header is 'image/png', otherwise JSON",
        "content": {
            "application/json": {
                "example": [{
                    "base64": "...very long string...",
                    "seed": "1050625087",
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


def get_task_type(req: Text2ImgRequest) -> TaskType:
    if isinstance(req, ImgUpscaleOrVaryRequest) or isinstance(req, ImgUpscaleOrVaryRequestJson):
        return TaskType.img_uov
    elif isinstance(req, ImgPromptRequest) or isinstance(req, ImgPromptRequestJson):
        return TaskType.img_prompt
    elif isinstance(req, ImgInpaintOrOutpaintRequest) or isinstance(req, ImgInpaintOrOutpaintRequestJson):
        return TaskType.img_inpaint_outpaint
    else:
        return TaskType.text_2_img


def call_worker(req: Text2ImgRequest, accept: str) -> Response | AsyncJobResponse | List[GeneratedImageResult]:
    if accept == 'image/png':
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    task_type = get_task_type(req)
    params = req_to_params(req)
    async_task = worker_queue.add_task(task_type, params, req.webhook_url)

    if async_task is None:
        # add to worker queue failed
        failure_results = [ImageGenerationResult(im=None, seed='', finish_reason=GenerationFinishReason.queue_is_full)]

        if streaming_output:
            return generate_streaming_output(failure_results)
        if req.async_process:
            return AsyncJobResponse(job_id='',
                                    job_type=get_task_type(req),
                                    job_stage=AsyncJobStage.error,
                                    job_progress=0,
                                    job_status=None,
                                    job_step_preview=None,
                                    job_result=failure_results)
        else:
            return generate_image_result_output(failure_results, False)

    if req.async_process:
        # return async response directly
        return generate_async_output(async_task)
    
    # blocking get generation result
    results = blocking_get_task_result(async_task.job_id)

    if streaming_output:
        return generate_streaming_output(results)
    else:
        return generate_image_result_output(results, req.require_base64)


def stop_worker():
    process_top()


@app.get("/")
def home():
    return Response(content='Swagger-UI to: <a href="/docs">/docs</a>', media_type="text/html")


@app.get("/ping", description="Returns a simple 'pong' response")
def ping():
    return Response(content='pong', media_type="text/html")


@secure_router.post("/v1/generation/text-to-image", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def text2img_generation(req: Text2ImgRequest, accept: str = Header(None),
                        accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post("/v2/generation/text-to-image-with-ip", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def text_to_img_with_ip(req: Text2ImgRequestWithPrompt,
                        accept: str = Header(None),
                        accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for img_prompt in req.image_prompts:
        img_prompt.cn_img = base64_to_stream(img_prompt.cn_img)
        image = ImagePrompt(cn_img=img_prompt.cn_img,
                            cn_stop=img_prompt.cn_stop,
                            cn_weight=img_prompt.cn_weight,
                            cn_type=img_prompt.cn_type)
        image_prompts_files.append(image)

    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)

    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post("/v1/generation/image-upscale-vary", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def img_upscale_or_vary(input_image: UploadFile, req: ImgUpscaleOrVaryRequest = Depends(ImgUpscaleOrVaryRequest.as_form),
                        accept: str = Header(None),
                        accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post("/v2/generation/image-upscale-vary", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def img_upscale_or_vary_v2(req: ImgUpscaleOrVaryRequestJson,
                           accept: str = Header(None),
                           accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    req.input_image = base64_to_stream(req.input_image)

    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for img_prompt in req.image_prompts:
        img_prompt.cn_img = base64_to_stream(img_prompt.cn_img)
        image = ImagePrompt(cn_img=img_prompt.cn_img,
                            cn_stop=img_prompt.cn_stop,
                            cn_weight=img_prompt.cn_weight,
                            cn_type=img_prompt.cn_type)
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)
    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post("/v1/generation/image-inpaint-outpaint", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def img_inpaint_or_outpaint(input_image: UploadFile, req: ImgInpaintOrOutpaintRequest = Depends(ImgInpaintOrOutpaintRequest.as_form),
                            accept: str = Header(None),
                            accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post("/v2/generation/image-inpaint-outpaint", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def img_inpaint_or_outpaint_v2(req: ImgInpaintOrOutpaintRequestJson,
                               accept: str = Header(None),
                               accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)
    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for img_prompt in req.image_prompts:
        img_prompt.cn_img = base64_to_stream(img_prompt.cn_img)
        image = ImagePrompt(cn_img=img_prompt.cn_img,
                            cn_stop=img_prompt.cn_stop,
                            cn_weight=img_prompt.cn_weight,
                            cn_type=img_prompt.cn_type)
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)
    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post("/v1/generation/image-prompt", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def img_prompt(cn_img1: Optional[UploadFile] = File(None),
               req: ImgPromptRequest = Depends(ImgPromptRequest.as_form),
               accept: str = Header(None),
               accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post("/v2/generation/image-prompt", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def img_prompt(req: ImgPromptRequestJson,
               accept: str = Header(None),
               accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if req.input_image is not None:
        req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)

    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for img_prompt in req.image_prompts:
        img_prompt.cn_img = base64_to_stream(img_prompt.cn_img)
        image = ImagePrompt(cn_img=img_prompt.cn_img,
                            cn_stop=img_prompt.cn_stop,
                            cn_weight=img_prompt.cn_weight,
                            cn_type=img_prompt.cn_type)
        image_prompts_files.append(image)

    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)

    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.get("/v1/generation/query-job", response_model=AsyncJobResponse, description="Query async generation job")
def query_job(req: QueryJobRequest = Depends()):
    queue_task = worker_queue.get_task(req.job_id, True)
    if queue_task is None:
        result = AsyncJobResponse(job_id="",
                                 job_type=TaskType.not_found,
                                 job_stage=AsyncJobStage.error,
                                 job_progress=0,
                                 job_status="Job not found")
        content = result.model_dump_json()
        return Response(content=content, media_type='application/json', status_code=404)
    return generate_async_output(queue_task, req.require_step_preview)


@secure_router.get("/v1/generation/job-queue", response_model=JobQueueInfo, description="Query job queue info")
def job_queue():
    return JobQueueInfo(running_size=len(worker_queue.queue), finished_size=len(worker_queue.history), last_job_id=worker_queue.last_job_id)


@secure_router.get("/v1/generation/job-history", response_model=JobHistoryResponse | dict, description="Query historical job data")
def get_history(job_id: str = None, page: int = 0, page_size: int = 20):
    # Fetch and return the historical tasks
    queue = [JobHistoryInfo(job_id=item.job_id, is_finished=item.is_finished) for item in worker_queue.queue]
    if not args.persistent:
        history = [JobHistoryInfo(job_id=item.job_id, is_finished=item.is_finished) for item in worker_queue.history]
        return JobHistoryResponse(history=history, queue=queue)
    else:
        from fooocusapi.sql_client import query_history
        history = query_history(task_id=job_id, page=page, page_size=page_size)
        return {
            "history": history,
            "queue": queue
        }


@secure_router.post("/v1/generation/stop", response_model=StopResponse, description="Job stoping")
def stop():
    stop_worker()
    return StopResponse(msg="success")


@secure_router.post("/v1/tools/describe-image", response_model=DescribeImageResponse)
def describe_image(image: UploadFile, type: DescribeImageType = Query(DescribeImageType.photo, description="Image type, 'Photo' or 'Anime'")):
    if type == DescribeImageType.photo:
        from extras.interrogate import default_interrogator as default_interrogator_photo
        interrogator = default_interrogator_photo
    else:
        from extras.wd14tagger import default_interrogator as default_interrogator_anime
        interrogator = default_interrogator_anime
    img = HWC3(read_input_image(image))
    result = interrogator(img)
    return DescribeImageResponse(describe=result)


@secure_router.get("/v1/engines/all-models", response_model=AllModelNamesResponse, description="Get all filenames of base model and lora")
def all_models():
    from modules import config
    config.update_files()
    return AllModelNamesResponse(model_filenames=config.model_filenames, lora_filenames=config.lora_filenames)


@secure_router.post("/v1/engines/refresh-models", response_model=AllModelNamesResponse, description="Features are merged into all_models, and the interface will be removed later", deprecated=True)
def refresh_models():
    from modules import config
    config.update_files()
    return AllModelNamesResponse(model_filenames=config.model_filenames, lora_filenames=config.lora_filenames)


@secure_router.get("/v1/engines/styles", response_model=List[str], description="Get all legal Fooocus styles")
def all_styles():
    from modules.sdxl_styles import legal_style_names
    return legal_style_names


app.mount("/files", StaticFiles(directory=file_utils.output_dir), name="files")

app.include_router(secure_router)

def start_app(args):
    file_utils.static_serve_base_url = args.base_url + "/files/"
    uvicorn.run("fooocusapi.api:app", host=args.host,
                port=args.port, log_level=args.log_level)
