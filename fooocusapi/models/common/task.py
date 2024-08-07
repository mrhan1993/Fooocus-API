"""
Task and job related models
"""
from enum import Enum
from pydantic import (
    BaseModel,
    Field
)


class TaskType(str, Enum):
    """
    Task type object
    """
    text_2_img = 'Text to Image'
    img_uov = 'Image Upscale or Variation'
    img_inpaint_outpaint = 'Image Inpaint or Outpaint'
    img_prompt = 'Image Prompt'
    img_enhance = 'Image Enhancement'
    not_found = 'Not Found'


class GenerationFinishReason(str, Enum):
    """
    Generation finish reason
    """
    success = 'SUCCESS'
    queue_is_full = 'QUEUE_IS_FULL'
    user_cancel = 'USER_CANCEL'
    error = 'ERROR'


class ImageGenerationResult:
    """
    Image generation result
    """
    def __init__(self, im: str | None, seed: str, finish_reason: GenerationFinishReason):
        self.im = im
        self.seed = seed
        self.finish_reason = finish_reason


class AsyncJobStage(str, Enum):
    """
    Async job stage
    """
    waiting = 'WAITING'
    running = 'RUNNING'
    success = 'SUCCESS'
    error = 'ERROR'


class GeneratedImageResult(BaseModel):
    """
    Generated images result
    """
    base64: str | None = Field(
        description="Image encoded in base64, or null if finishReason is not 'SUCCESS', only return when request require base64")
    url: str | None = Field(description="Image file static serve url, or null if finishReason is not 'SUCCESS'")
    seed: str = Field(description="The seed associated with this image")
    finish_reason: GenerationFinishReason
