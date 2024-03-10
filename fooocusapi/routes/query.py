"""Query API"""
from typing import List
from fastapi import APIRouter, Depends, Response
from fooocusapi.models.common.response import AllModelNamesResponse
from fooocusapi.utils.api_utils import api_key_auth
from fooocusapi.tasks.task_queue import task_queue

secure_router = APIRouter(dependencies=[Depends(api_key_auth)])


@secure_router.get("/")
def home():
    """Home page"""
    return Response(
        content='Swagger-UI to: <a href="/docs">/docs</a>',
        media_type="text/html"
    )


@secure_router.get(path="/ping", description="Returns a simple 'pong' response")
async def ping():
    """
    Ping page, just to check if the fastapi is up.
    Instant return correct, does not mean the service is available.
    Returns:
         A simple string Pong
    """
    return 'Pong'


@secure_router.get(
    path="/v1/generation/task/{task_id}",
    description="Query task info by task id",
    tags=['query job']
)
async def query_job(task_id: str):
    """
    Query job status use job id, return None if not found.
    Args:
        task_id: string task id
    Returns:
        Task info, a dict
    """
    if task_queue.current is not None and task_id == task_queue.current.task_id:
        return task_queue.current.to_dict()
    for task in task_queue.history:
        if task.task_id == task_id:
            return task.to_dict()
    for task in task_queue.queue._queue:
        if task.task_id == task_id:
            return task.to_dict()
    return None


@secure_router.get(
    path="/v1/generation/job-queue",
    description="Query job queue info",
    tags=['query job']
)
async def job_queue():
    """
    Get all tasks, including running, history and waiting tasks.
    Returns:
        A dict, include running, history and waiting tasks.
    Example:
        {
            'pending': list[TaskInfo],
            'running': TaskInfo,
            'history': list[TaskInfo]
        }
    """
    if len(task_queue.queue._queue) == 0:
        pending = []
    else:
        pending = [task.to_dict() for task in task_queue.queue._queue]
    running = task_queue.current.to_dict() if task_queue.current is not None else None
    if len(task_queue.history) == 0:
        history = []
    else:
        history = [task.to_dict() for task in task_queue.history]
    tasks = {
        'pending': pending,
        'running': running,
        'history': history
    }
    return tasks


@secure_router.get(
    path="/v1/engines/all-models",
    response_model=AllModelNamesResponse,
    description="Refresh local files and Get all filenames of base model and lora",
)
async def all_models():
    """
    Refresh local files and get all filenames of base model and lora
    """
    from modules import config

    config.update_all_model_names()
    return AllModelNamesResponse(
        model_filenames=config.model_filenames,
        lora_filenames=config.lora_filenames
    )


@secure_router.get(
    path="/v1/engines/styles",
    response_model=List[str],
    description="Get all legal Fooocus styles",
)
async def all_styles():
    """
    Get all legal Fooocus styles
    Returns:
        A list of strings, each string is a legal Fooocus style.
    """
    from modules.sdxl_styles import legal_style_names

    return legal_style_names
