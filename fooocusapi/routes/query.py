"""Query API"""
from typing import List
from fastapi import APIRouter, Depends, Response
from fooocusapi.args import args
from fooocusapi.models.common.requests import QueryJobRequest
from fooocusapi.models.common.response import (
    AllModelNamesResponse,
    AsyncJobResponse,
    AsyncJobStage,
    JobHistoryInfo,
    JobHistoryResponse,
    JobQueueInfo,
)
from fooocusapi.task_queue import TaskType
from fooocusapi.utils.api_utils import api_key_auth, generate_async_output
from fooocusapi.worker import worker_queue


secure_router = APIRouter(dependencies=[Depends(api_key_auth)])

@secure_router.get("/")
def home():
    """Home page"""
    return Response(
        content='Swagger-UI to: <a href="/docs">/docs</a>',
        media_type="text/html"
    )


@secure_router.get("/ping", description="Returns a simple 'pong' response")
def ping():
    """Ping page"""
    return Response(
        content="pong",
        media_type="text/html"
    )


@secure_router.get(
    "/v1/generation/query-job",
    response_model=AsyncJobResponse,
    description="Query async generation job",
)
def query_job(req: QueryJobRequest = Depends()):
    """Query job status use job id"""
    queue_task = worker_queue.get_task(req.job_id, True)
    if queue_task is None:
        result = AsyncJobResponse(
            job_id="",
            job_type=TaskType.not_found,
            job_stage=AsyncJobStage.error,
            job_progress=0,
            job_status="Job not found",
        )
        content = result.model_dump_json()
        return Response(content=content, media_type="application/json", status_code=404)
    return generate_async_output(queue_task, req.require_step_preview)


@secure_router.get(
    "/v1/generation/job-queue",
    response_model=JobQueueInfo,
    description="Query job queue info",
)
def job_queue():
    """Get job queue info"""
    return JobQueueInfo(
        running_size=len(worker_queue.queue),
        finished_size=len(worker_queue.history),
        last_job_id=worker_queue.last_job_id,
    )


@secure_router.get(
    "/v1/generation/job-history",
    response_model=JobHistoryResponse | dict,
    description="Query historical job data",
)
def get_history(job_id: str = None, page: int = 0, page_size: int = 20):
    """Fetch and return the historical tasks"""
    queue = [
        JobHistoryInfo(job_id=item.job_id, is_finished=item.is_finished)
        for item in worker_queue.queue
    ]
    if not args.persistent:
        history = [
            JobHistoryInfo(job_id=item.job_id, is_finished=item.is_finished)
            for item in worker_queue.history
        ]
        return JobHistoryResponse(history=history, queue=queue)
    else:
        from fooocusapi.sql_client import query_history

        history = query_history(task_id=job_id, page=page, page_size=page_size)
        return {"history": history, "queue": queue}


@secure_router.get(
    "/v1/engines/all-models",
    response_model=AllModelNamesResponse,
    description="Get all filenames of base model and lora",
)
def all_models():
    """Refresh local files and get all filenames of base model and lora"""
    from modules import config

    config.update_all_model_names()
    return AllModelNamesResponse(
        model_filenames=config.model_filenames,
        lora_filenames=config.lora_filenames
    )


@secure_router.get(
    "/v1/engines/styles",
    response_model=List[str],
    description="Get all legal Fooocus styles",
)
def all_styles():
    """Get all legal Fooocus styles"""
    from modules.sdxl_styles import legal_style_names

    return legal_style_names
