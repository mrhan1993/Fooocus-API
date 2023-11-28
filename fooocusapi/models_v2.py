from fooocusapi.models import *


class ImgUpscaleOrVaryRequestJson(Text2ImgRequest):
    uov_method: UpscaleOrVaryMethod = "Upscale (2x)"
    input_image: str = Field(description="Init image for upsacale or outpaint as base64")


class ImgInpaintOrOutpaintRequestJson(Text2ImgRequest):
    input_image: str = Field(description="Init image for inpaint or outpaint as base64")
    input_mask: str | None = Field(None, description="Inpaint or outpaint mask as base64")
    inpaint_additional_prompt: str | None = Field(None, description="Describe what you want to inpaint")
    outpaint_selections: List[OutpaintExpansion] = []
    outpaint_distance_left: int = Field(default=0, description="Set outpaint left distance"),
    outpaint_distance_right: int = Field(default=0, description="Set outpaint right distance"),
    outpaint_distance_top: int = Field(default=0, description="Set outpaint top distance"),
    outpaint_distance_bottom: int = Field(default=0, description="Set outpaint bottom distance"),


class ImagePromptJson(BaseModel):
    cn_img: str | None = Field(None, description="Input image for image prompt as base64")
    cn_stop: float | None = Field(None, ge=0, le=1, description="Stop at for image prompt, None for default value")
    cn_weight: float | None = Field(None, ge=0, le=2, description="Weight for image prompt, None for default value")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip, description="ControlNet type for image prompt")


class ImgPromptRequestJson(Text2ImgRequest):
    image_prompts: List[ImagePromptJson | ImagePrompt]
