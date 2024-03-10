"""Task object models"""
from enum import Enum
from pydantic import (
    BaseModel,
    Field
)


class AsyncJobStage(str, Enum):
    """Async job stage"""
    waiting = 'WAITING'
    running = 'RUNNING'
    success = 'SUCCESS'
    error = 'ERROR'


class TaskType(str, Enum):
    """Task type"""
    text_2_img = 'Text to Image'
    img_uov = 'Image Upscale or Variation'
    img_inpaint_outpaint = 'Image Inpaint or Outpaint'
    img_prompt = 'Image Prompt'
    not_found = 'Not Found'


class GenerationFinishReason(str, Enum):
    """Generation finish reason"""
    success = 'SUCCESS'
    queue_is_full = 'QUEUE_IS_FULL'
    user_cancel = 'USER_CANCEL'
    error = 'ERROR'


class GeneratedImageResult(BaseModel):
    """Generated image result"""
    base64: str | None = Field(
        description="Null or base64 image, only return when request require base64")
    url: str | None = Field(description="Image file static serve url, or null")
    seed: str = Field(description="The seed associated with this image")
    finish_reason: GenerationFinishReason = Field(
        default=GenerationFinishReason.success, description="Reason of generation finish")
