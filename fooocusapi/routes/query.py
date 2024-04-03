"""Query API"""
from typing import List
from fastapi import Depends, Response, APIRouter

from fooocusapi.args import args

from fooocusapi.models.common.requests import QueryJobRequest
from fooocusapi.models.common.response import (
    AsyncJobResponse,
    JobHistoryInfo,
    JobQueueInfo,
    JobHistoryResponse,
    AllModelNamesResponse
)
from fooocusapi.models.common.task import AsyncJobStage

from fooocusapi.utils.api_utils import generate_async_output, api_key_auth
from fooocusapi.task_queue import TaskType
from fooocusapi.worker import worker_queue

secure_router = APIRouter(dependencies=[Depends(api_key_auth)])


@secure_router.get(path="/", tags=['Query'])
def home():
    """Home page"""
    return Response(
        content='Swagger-UI to: <a href="/docs">/docs</a>',
        media_type="text/html"
    )


@secure_router.get(
        path="/ping",
        description="Returns a simple 'pong'",
        tags=['Query'])
async def ping():
    """\nPing\n
    Ping page, just to check if the fastapi is up.
    Instant return correct, does not mean the service is available.
    Returns:
        A simple string pong
    """
    return 'pong'


@secure_router.get(
        path="/v1/generation/query-job",
        response_model=AsyncJobResponse,
        description="Query async generation job",
        tags=['Query'])
def query_job(req: QueryJobRequest = Depends()):
    """query job info by id"""
    queue_task = worker_queue.get_task(req.job_id, True)
    if queue_task is None:
        result = AsyncJobResponse(
            job_id="",
            job_type=TaskType.not_found,
            job_stage=AsyncJobStage.error,
            job_progress=0,
            job_status="Job not found")
        content = result.model_dump_json()
        return Response(content=content, media_type='application/json', status_code=404)
    return generate_async_output(queue_task, req.require_step_preview)


@secure_router.get(
        path="/v1/generation/job-queue",
        response_model=JobQueueInfo,
        description="Query job queue info",
        tags=['Query'])
def job_queue():
    """Query job queue info"""
    queue = JobQueueInfo(
        running_size=len(worker_queue.queue),
        finished_size=len(worker_queue.history),
        last_job_id=worker_queue.last_job_id
    )
    return queue


@secure_router.get(
        path="/v1/generation/job-history",
        response_model=JobHistoryResponse | dict,
        description="Query historical job data",
        tags=["Query"])
def get_history(job_id: str = None, page: int = 0, page_size: int = 20):
    """Fetch and return the historical tasks"""
    queue = [
        JobHistoryInfo(
            job_id=item.job_id,
            is_finished=item.is_finished
        ) for item in worker_queue.queue
    ]
    if not args.persistent:
        history = [
            JobHistoryInfo(
                job_id=item.job_id,
                is_finished=item.is_finished
            ) for item in worker_queue.history
        ]
        return JobHistoryResponse(history=history, queue=queue)

    from fooocusapi.sql_client import query_history
    history = query_history(task_id=job_id, page=page, page_size=page_size)
    return {
        "history": history,
        "queue": queue
    }


@secure_router.get(
        path="/v1/engines/all-models",
        response_model=AllModelNamesResponse,
        description="Get all filenames of base model and lora",
        tags=["Query"])
def all_models():
    """Refresh and return all models"""
    from modules import config
    config.update_files()
    models = AllModelNamesResponse(
        model_filenames=config.model_filenames,
        lora_filenames=config.lora_filenames)
    return models


@secure_router.get(
        path="/v1/engines/styles",
        response_model=List[str],
        description="Get all legal Fooocus styles",
        tags=['Query'])
def all_styles():
    """Return all available styles"""
    from modules.sdxl_styles import legal_style_names
    return legal_style_names
