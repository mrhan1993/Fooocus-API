from pydantic import BaseModel, Field, field_validator
from typing import List
from enum import Enum


class AdvancedParams(BaseModel):
    """
    Advanced parameters for the API.
    """
    adm_scaler_positive: float = 1.5
    adm_scaler_negative: float = 0.6
    adm_scaler_end: float = 0.3
    refiner_swap_method: str = "joint"
    adaptive_cfg: int = 7
    sampler_name: str = "dpmpp_2m_sde_gpu"
    scheduler_name: str = "karras"
    overwrite_step: int = -1
    overwrite_switch: int = -1
    overwrite_width: int = -1
    overwrite_height: int = -1
    overwrite_vary_strength: int = -1
    overwrite_upscale_strength: int = -1
    mixing_image_prompt_and_vary_upscale: bool = False
    mixing_image_prompt_and_inpaint: bool = False
    debugging_cn_preprocessor: bool = False
    skipping_cn_preprocessor: bool = False
    controlnet_softness: float = 0.25
    canny_low_threshold: int = 64
    canny_high_threshold: int = 128
    inpaint_engine: str = 'v1'
    freeu_enabled : bool = False
    freeu_b1: float = 1.01
    freeu_b2: float = 1.02
    freeu_s1: float = 0.99
    freeu_s2: float = 0.95


class Lora(BaseModel):
    """
    Lora model.
    """
    model_name: str = "sd_xl_offset_example-lora_1.0.safetensors"
    weights: float = 0.1


class PerformanceEnum(str, Enum):
    """
    Performance enum.
    """
    speed = "Speed"
    quality = "Quality"
    extreme_speed = "Extreme Speed"

class UpscaleOrVaryMethod(str, Enum):
    """
    UovMethod enum.
    """
    subtle = "Vary (Subtle)"
    strong = "Vary (Strong)"
    upscale_15 = "Upscale (1.5x)"
    upscale_20 = "Upscale (2x)"
    upscale_fast = "Upscale (Fast 2x)"
    upscale_custom = "Upscale (Custom)"

class ControlNetEnum(str, Enum):
    """
    ControlNet enum.
    """
    imagePrompt = "ImagePrompt"
    faceSwap = "FaceSwap"
    pyraCanny = "PyraCanny"
    cpds = "CPDS"

class OutpaintExpansion(str, Enum):
    left = 'Left'
    right = 'Right'
    top = 'Top'
    bottom = 'Bottom'

class ImagePrompt(BaseModel):
    cn_img: str | None = None
    cn_stop: float | None = 0.6
    cn_weight: float | None = 0.6
    cn_type: ControlNetEnum = "ImagePrompt"

class SharedFields(BaseModel):
    """
    SharedFields model.
    """
    prompt: str = ""
    negative_prompt: str = ""
    style_selections: str = ""
    aspect_ratios_selection: str = "1152*896"
    performance_selection: PerformanceEnum = "Speed"
    image_number: int = 1
    image_seed: int = -1
    sharpness: int = 2
    guidance_scale: int = 4
    base_model_name: str = "juggernautXL_version6Rundiffusion.safetensors"
    refiner_model_name: str = "None"
    refiner_switch: float = 0.5
    loras: str = '[{"model_name":"sd_xl_offset_example-lora_1.0.safetensors","weight":0.1}]'
    advanced_params: str = ""
    require_base64: bool = False
    async_process: bool = False


class Text2ImgParams(SharedFields):
    """
    Text2ImgPrompt model.
    """
    style_selections: List[str] = ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"]
    loras: List[Lora] = [Lora()]
    advanced_params: AdvancedParams = AdvancedParams()

class ImgUpscaleOrVaryParams(SharedFields):
    """
    Upscale model.
    """
    uov_method: UpscaleOrVaryMethod = "Upscale (2x)"
    upscale_value: float | None = Field(None, ge=1.0, le=5.0, description="Upscale custom value, None for default value")

class ImgInpaintOrOutpaintParams(SharedFields):
    """
    Inpaint model.
    """
    outpaint_selections: str = Field("", description="A comma-separated string of 'Left', 'Right', 'Top', 'Bottom'")
    outpaint_distance_left: int = Field(default=0, description="Set outpaint left distance, 0 for default")
    outpaint_distance_right: int = Field(default=0, description="Set outpaint right distance, 0 for default")
    outpaint_distance_top: int = Field(default=0, description="Set outpaint top distance, 0 for default")
    outpaint_distance_bottom: int = Field(default=0, description="Set outpaint bottom distance, 0 for default")

    @field_validator("outpaint_selections")
    def validate_outpaint_selections(cls, v):
        if v == "":
            return v
        allowed_values = ['Left', 'Right', 'Top', 'Bottom']
        for val in v.split(','):
            if val not in allowed_values:
                raise ValueError(f"Value {val} is invalid, allowed values are {allowed_values}")
        return v


class ImagePromptParams(SharedFields):
    """
    Image prompt.
    """
    cn_stop1: float = 0.6
    cn_weight1: float = 0.6
    cn_type1: ControlNetEnum = "ImagePrompt"
    cn_stop2: float = 0.6
    cn_weight2: float = 0.6
    cn_type2: ControlNetEnum = "ImagePrompt"
    cn_stop3: float = 0.6
    cn_weight3: float = 0.6
    cn_type3: ControlNetEnum = "ImagePrompt"
    cn_stop4: float = 0.6
    cn_weight4: float = 0.6
    cn_type4: ControlNetEnum = "ImagePrompt"


class ImgUpscaleOrVaryParamsJson(Text2ImgParams):
    uov_method: UpscaleOrVaryMethod = "Upscale (2x)"
    input_image: str = Field(description="Init image for upsacale or outpaint as base64")

class ImgInpaintOrOutpaintParamsJson(Text2ImgParams):
    input_image: str
    input_mask: str | None = None
    inpaint_additional_prompt: str | None = None
    outpaint_selections: List[OutpaintExpansion] = ["Left", "Right", "Top", "Bottom"]
    outpaint_distance_left: int = Field(default=0, description="Set outpaint left distance, 0 for default")
    outpaint_distance_right: int = Field(default=0, description="Set outpaint right distance, 0 for default")
    outpaint_distance_top: int = Field(default=0, description="Set outpaint top distance, 0 for default")
    outpaint_distance_bottom: int = Field(default=0, description="Set outpaint bottom distance, 0 for default")

class ImagePromptParamsJson(Text2ImgParams):
    image_prompts: List[ImagePrompt] = [ImagePrompt()]
