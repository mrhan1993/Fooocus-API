import uvicorn

from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi import Depends, FastAPI, Header, Query, Response, UploadFile
from fastapi.params import File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from fooocusapi.models import *
from fooocusapi.api_utils import generation_output, req_to_params
import fooocusapi.file_utils as file_utils
from fooocusapi.parameters import GenerationFinishReason, ImageGenerationResult
from fooocusapi.task_queue import TaskType
from fooocusapi.worker import process_generate, task_queue, process_top
from fooocusapi.models_v2 import *
from fooocusapi.img_utils import base64_to_stream, read_input_image

from concurrent.futures import ThreadPoolExecutor
from modules.util import HWC3


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源访问
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

work_executor = ThreadPoolExecutor(
    max_workers=task_queue.queue_size*2, thread_name_prefix="worker_")

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


def call_worker(req: Text2ImgRequest, accept: str):
    task_type = TaskType.text_2_img
    if isinstance(req, ImgUpscaleOrVaryRequest) or isinstance(req, ImgUpscaleOrVaryRequestJson):
        task_type = TaskType.img_uov
    elif isinstance(req, ImgPromptRequest) or isinstance(req, ImgPromptRequestJson):
        task_type = TaskType.img_prompt
    elif isinstance(req, ImgInpaintOrOutpaintRequest) or isinstance(req, ImgInpaintOrOutpaintRequestJson):
        task_type = TaskType.img_inpaint_outpaint

    params = req_to_params(req)
    queue_task = task_queue.add_task(
        task_type, {'params': params.__dict__, 'accept': accept, 'require_base64': req.require_base64},
        webhook_url=req.webhook_url)

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

def stop_worker():
    process_top()

@app.get("/")
def home():
    return Response(content='Swagger-UI to: <a href="/docs">/docs</a>', media_type="text/html")


@app.get("/ping", description="Returns a simple 'pong' response")
def ping():
    return Response(content='pong', media_type="text/html")

@app.post("/v1/generation/text-to-image", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
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
    return generation_output(results, streaming_output, req.require_base64)

@app.post("/v2/generation/text-to-image-with-ip", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def text_to_img_with_ip(req: Text2ImgRequestWithPrompt,
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

    results = call_worker(req, accept)
    return generation_output(results, streaming_output, req.require_base64)

@app.post("/v1/generation/image-upscale-vary", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
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
    return generation_output(results, streaming_output, req.require_base64)


@app.post("/v2/generation/image-upscale-vary", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def img_upscale_or_vary_v2(req: ImgUpscaleOrVaryRequestJson,
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
    req.input_image = base64_to_stream(req.input_image)

    results = call_worker(req, accept)
    return generation_output(results, streaming_output, req.require_base64)


@app.post("/v1/generation/image-inpait-outpaint", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
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
    return generation_output(results, streaming_output, req.require_base64)


@app.post("/v2/generation/image-inpait-outpaint", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def img_inpaint_or_outpaint_v2(req: ImgInpaintOrOutpaintRequestJson,
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

    req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)
    results = call_worker(req, accept)
    return generation_output(results, streaming_output, req.require_base64)


@app.post("/v1/generation/image-prompt", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
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
    return generation_output(results, streaming_output, req.require_base64)


@app.post("/v2/generation/image-prompt", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def img_prompt(req: ImgPromptRequestJson,
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

    results = call_worker(req, accept)
    return generation_output(results, streaming_output, req.require_base64)


@app.get("/v1/generation/query-job", response_model=AsyncJobResponse, description="Query async generation job")
def query_job(req: QueryJobRequest = Depends()):
    queue_task = task_queue.get_task(req.job_id, True)
    if queue_task is None:
        return JSONResponse(content=AsyncJobResponse(job_id="",
                                                     job_type="Not Found",
                                                     job_stage="ERROR",
                                                     job_progress=0,
                                                     job_status="Job not found"), status_code=404)

    return generation_output(queue_task, streaming_output=False, require_base64=False,
                             require_step_preivew=req.require_step_preivew)


@app.get("/v1/generation/job-queue", response_model=JobQueueInfo, description="Query job queue info")
def job_queue():
    return JobQueueInfo(running_size=len(task_queue.queue), finished_size=len(task_queue.history), last_job_id=task_queue.last_job_id)


@app.get("/v1/generation/job-history", response_model=JobHistoryResponse, description="Query historical job data")
def get_history():
    # Fetch and return the historical tasks
    hitory = [JobHistoryInfo(job_id=item.job_id, is_finished=item.is_finished) for item in task_queue.history]
    queue = [JobHistoryInfo(job_id=item.job_id, is_finished=item.is_finished) for item in task_queue.queue]
    return JobHistoryResponse(history=hitory, queue=queue)


@app.post("/v1/generation/stop", response_model=StopResponse, description="Job stoping")
def stop():
    stop_worker()
    return StopResponse(msg="success")


@app.post("/v1/tools/describe-image", response_model=DescribeImageResponse)
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


@app.get("/v1/engines/all-models", response_model=AllModelNamesResponse, description="Get all filenames of base model and lora")
def all_models():
    import modules.config as config
    return AllModelNamesResponse(model_filenames=config.model_filenames, lora_filenames=config.lora_filenames)


@app.post("/v1/engines/refresh-models", response_model=AllModelNamesResponse, description="Refresh local files and get all filenames of base model and lora")
def refresh_models():
    import modules.config as config
    config.update_all_model_names()
    return AllModelNamesResponse(model_filenames=config.model_filenames, lora_filenames=config.lora_filenames)


@app.get("/v1/engines/styles", response_model=List[str], description="Get all legal Fooocus styles")
def all_styles():
    from modules.sdxl_styles import legal_style_names
    return legal_style_names


app.mount("/files", StaticFiles(directory=file_utils.output_dir), name="files")


def start_app(args):
    file_utils.static_serve_base_url = args.base_url + "/files/"
    uvicorn.run("fooocusapi.api:app", host=args.host,
                port=args.port, log_level=args.log_level)
