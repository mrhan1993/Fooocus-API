"""V2 API models"""
from typing import List
from pydantic import BaseModel, Field
from fooocusapi.models.common.base import (
    CommonRequest as Text2ImgRequest,
    ControlNetType,
    OutpaintExpansion,
    ImagePrompt,
    UpscaleOrVaryMethod
)

class ImagePromptJson(BaseModel):
    """Image prompt"""
    cn_img: str | None = Field(None,
                               description="Input image for image prompt as base64")
    cn_stop: float | None = Field(0, ge=0, le=1,
                                  description="Stop at for image prompt, 0 for default value")
    cn_weight: float | None = Field(0, ge=0, le=2,
                                    description="Weight for image prompt, 0 for default value")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip,
                                    description="ControlNet type for image prompt")


class ImgUpscaleOrVaryRequestJson(Text2ImgRequest):
    """Image inpaint or outpaint request"""
    uov_method: UpscaleOrVaryMethod = "Upscale (2x)"
    upscale_value: float | None = Field(1.0, ge=1.0, le=5.0,
                                        description="Upscale custom value, 1.0 for default value")
    input_image: str = Field(description="Init image for upsacale or outpaint as base64")
    image_prompts: List[ImagePromptJson | ImagePrompt] = []


class ImgInpaintOrOutpaintRequestJson(Text2ImgRequest):
    """Image inpaint or outpaint request"""
    input_image: str = Field(description="Init image for inpaint or outpaint as base64")
    input_mask: str | None = Field('',
                                   description="Inpaint or outpaint mask as base64")
    inpaint_additional_prompt: str | None = Field('',
                                                  description="Describe what you want to inpaint")
    outpaint_selections: List[OutpaintExpansion] = []
    outpaint_distance_left: int | None = Field(-1,
                                               description="Set outpaint left distance")
    outpaint_distance_right: int | None = Field(-1,
                                                description="Set outpaint right distance")
    outpaint_distance_top: int | None = Field(-1,
                                              description="Set outpaint top distance")
    outpaint_distance_bottom: int | None = Field(-1,
                                                 description="Set outpaint bottom distance")
    image_prompts: List[ImagePromptJson | ImagePrompt] = []


class ImgPromptRequestJson(ImgInpaintOrOutpaintRequestJson):
    """Image prompt request"""
    input_image: str | None = Field(None,
                                    description="Init image for inpaint or outpaint as base64")
    image_prompts: List[ImagePromptJson | ImagePrompt]


class Text2ImgRequestWithPrompt(Text2ImgRequest):
    """Text2Img request with image prompt"""
    image_prompts: List[ImagePromptJson] = []
