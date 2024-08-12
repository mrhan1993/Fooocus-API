"""Fooocus API models for response"""
from typing import List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field
)

from fooocusapi.models.common.task import (
    GeneratedImageResult,
    AsyncJobStage
)
from fooocusapi.task_queue import TaskType


class DescribeImageResponse(BaseModel):
    """
    describe image response
    """
    describe: str


class AsyncJobResponse(BaseModel):
    """
    Async job response
    Attributes:
        job_id: Job ID
        job_type: Job type
        job_stage: Job stage
        job_progress: Job progress, 0-100
        job_status: Job status
        job_step_preview: Job step preview
        job_result: Job result
    """
    job_id: str = Field(description="Job ID")
    job_type: TaskType = Field(description="Job type")
    job_stage: AsyncJobStage = Field(description="Job running stage")
    job_progress: int = Field(description="Job running progress, 100 is for finished.")
    job_status: str | None = Field(None, description="Job running status in text")
    job_step_preview: str | None = Field(None, description="Preview image of generation steps at current time, as base64 image")
    job_result: List[GeneratedImageResult] | None = Field(None, description="Job generation result")


class JobQueueInfo(BaseModel):
    """
    job queue info
    Attributes:
        running_size: int, The current running and waiting job count
        finished_size: int, The current finished job count
        last_job_id: str, Last submit generation job id
    """
    running_size: int = Field(description="The current running and waiting job count")
    finished_size: int = Field(description="Finished job count (after auto clean)")
    last_job_id: str | None = Field(description="Last submit generation job id")


# TODO May need more detail fields, will add later when someone need
class JobHistoryInfo(BaseModel):
    """
    job history info
    """
    job_id: str
    in_queue_mills: int
    start_mills: int
    finish_mills: int
    is_finished: bool = False


# Response model for the historical tasks
class JobHistoryResponse(BaseModel):
    """
    job history response
    """
    queue: List[JobHistoryInfo] = []
    history: List[JobHistoryInfo] = []


class AllModelNamesResponse(BaseModel):
    """
    all model list response
    """
    model_filenames: List[str] = Field(description="All available model filenames")
    lora_filenames: List[str] = Field(description="All available lora filenames")

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )


class StopResponse(BaseModel):
    """stop task response"""
    msg: str
