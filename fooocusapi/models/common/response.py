"""Fooocus API models for response"""
from typing import List
from enum import Enum
from pydantic import (
    BaseModel,
    ConfigDict,
    Field
)
from fooocusapi.task_queue import TaskType
from fooocusapi.parameters import GenerationFinishReason


class GeneratedImageResult(BaseModel):
    """Generated image result"""
    base64: str | None = Field(
        description="Null or base64 image, only return when request require base64")
    url: str | None = Field(description="Image file static serve url, or null")
    seed: str = Field(description="The seed associated with this image")
    finish_reason: GenerationFinishReason


class DescribeImageResponse(BaseModel):
    """Describe image result"""
    describe: str


class AsyncJobStage(str, Enum):
    """Async job stage"""
    waiting = 'WAITING'
    running = 'RUNNING'
    success = 'SUCCESS'
    error = 'ERROR'


class AsyncJobResponse(BaseModel):
    """"Async job response"""
    job_id: str = Field(default='', description="Job ID")
    job_type: TaskType = Field(description="Job type")
    job_stage: AsyncJobStage = Field(description="Job running stage")
    job_progress: int = Field(description="Job running progress, 100 is for finished.")
    job_status: str | None = Field(None, description="Job running status in text")
    job_step_preview: str | None = Field(
        default=None,
        description="Preview base64 image of generation steps at current time")
    job_result: List[GeneratedImageResult] | None = Field(None, description="Job generation result")


class JobQueueInfo(BaseModel):
    """Job queue info"""
    running_size: int = Field(description="The current running and waiting job count")
    finished_size: int = Field(description="Finished job cound (after auto clean)")
    last_job_id: str | None = Field(description="Last submit generation job id")


# TODO May need more detail fields, will add later when someone need
class JobHistoryInfo(BaseModel):
    """Job history info"""
    job_id: str
    is_finished: bool = False


# Response model for the historical tasks
class JobHistoryResponse(BaseModel):
    """Job history info"""
    queue: List[JobHistoryInfo] = []
    history: List[JobHistoryInfo] = []


class AllModelNamesResponse(BaseModel):
    """All model names"""
    model_filenames: List[str] = Field(description="All available model filenames")
    lora_filenames: List[str] = Field(description="All available lora filenames")

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )


class StopResponse(BaseModel):
    """Stop response"""
    msg: str
