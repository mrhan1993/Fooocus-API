from fooocusapi.models import *


class ImgUpscaleOrVaryRequestJson(Text2ImgRequest):
    uov_method: UpscaleOrVaryMethod = "Upscale (2x)"
    input_image: str = None


class ImgInpaintOrOutpaintRequestJson(Text2ImgRequest):
    input_image: str
    input_mask: str | None
    inpaint_additional_prompt: str | None
    outpaint_selections: List[OutpaintExpansion] = []


class ImagePromptJson(BaseModel):
    cn_img: str | None
    cn_stop: float = Field(default=0.6, ge=0, le=1)
    cn_weight: float | None = Field(
        default=0.6, ge=0, le=2, description="None for default value")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip)


class ImgPromptRequestJson(Text2ImgRequest):
    image_prompts: List[ImagePromptJson | ImagePrompt]
