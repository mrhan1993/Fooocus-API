"""Common modell for requests"""
from enum import Enum

from pydantic import BaseModel, Field


class DescribeImageType(str, Enum):
    """Image type for image to prompt"""
    photo = 'Photo'
    anime = 'Anime'


class QueryJobRequest(BaseModel):
    """Query job request"""
    job_id: str = Field(description="Job ID to query")
    require_step_preview: bool = Field(
        default=False,
        description="Set to true will return preview image of generation steps at current time")
