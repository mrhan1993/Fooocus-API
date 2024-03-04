class GeneratedImageResult(BaseModel):
    base64: str | None = Field(
        description="Image encoded in base64, or null if finishReasen is not 'SUCCESS', only return when request require base64")
    url: str | None = Field(description="Image file static serve url, or null if finishReasen is not 'SUCCESS'")
    seed: str = Field(description="The seed associated with this image")
    finish_reason: GenerationFinishReason


class DescribeImageResponse(BaseModel):
    describe: str


class AsyncJobStage(str, Enum):
    waiting = 'WAITING'
    running = 'RUNNING'
    success = 'SUCCESS'
    error = 'ERROR'


class QueryJobRequest(BaseModel):
    job_id: str = Field(description="Job ID to query")
    require_step_preview: bool = Field(False, description="Set to true will return preview image of generation steps at current time")


class AsyncJobResponse(BaseModel):
    job_id: str = Field(default='', description="Job ID")
    job_type: TaskType = Field(description="Job type")
    job_stage: AsyncJobStage = Field(description="Job running stage")
    job_progress: int = Field(description="Job running progress, 100 is for finished.")
    job_status: str | None = Field(None, description="Job running status in text")
    job_step_preview: str | None = Field(None, description="Preview image of generation steps at current time, as base64 image")
    job_result: List[GeneratedImageResult] | None = Field(None, description="Job generation result")


class JobQueueInfo(BaseModel):
    running_size: int = Field(description="The current running and waiting job count")
    finished_size: int = Field(description="Finished job cound (after auto clean)")
    last_job_id: str | None = Field(description="Last submit generation job id")


# TODO May need more detail fields, will add later when someone need
class JobHistoryInfo(BaseModel):
    job_id: str
    is_finished: bool = False


# Response model for the historical tasks
class JobHistoryResponse(BaseModel):
    queue: List[JobHistoryInfo] = []
    history: List[JobHistoryInfo] = []


class AllModelNamesResponse(BaseModel):
    model_filenames: List[str] = Field(description="All available model filenames")
    lora_filenames: List[str] = Field(description="All available lora filenames")

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )

    
class StopResponse(BaseModel):
    msg: str
