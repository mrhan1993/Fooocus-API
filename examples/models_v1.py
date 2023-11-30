from pydantic import BaseModel, Field, ValidationError, field_validator
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

class ControlNetEnum(str, Enum):
    """
    ControlNet enum.
    """
    imagePrompt = "ImagePrompt"
    faceSwap = "FaceSwap"
    pyraCanny = "PyraCanny"
    cpds = "CPDS"


class Text2ImgParams(BaseModel):
    """
    Text2ImgPrompt model.
    """
    prompt: str
    negative_prompt: str = ""
    style_selections: List[str] = ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"]
    performance_selection: PerformanceEnum = "Speed"
    aspect_ratios_selection: str = "1152×896"
    image_number: int = 1
    image_seed: int = -1
    sharpness: int = 2
    guidance_scale: int = 4
    base_model_name: str = "juggernautXL_version6Rundiffusion.safetensors"
    refiner_model_name: str = "None"
    refiner_switch: float = 0.5
    loras: List[Lora] = [Lora()]
    advanced_params: AdvancedParams = AdvancedParams()
    require_base64: bool = True
    async_process: bool = True


class SharedFields(BaseModel):
    """
    SharedFields model.
    """
    prompt: str = ""
    negative_prompt: str = ""
    style_selections: str = ""
    performance_selection: PerformanceEnum = "Speed"
    image_number: int = 1
    image_seed: int = -1
    sharpness: int = 2
    guidance_scale: int = 4
    base_model_name: str = "juggernautXL_version6Rundiffusion.safetensors"
    refiner_model_name: str = ""
    refiner_switch: float = 0.5
    loras: str = '[{"model_name":"sd_xl_offset_example-lora_1.0.safetensors","weight":0.1}]'
    advanced_params: str = ""
    require_base64: bool = True
    async_process: bool = True


class ImgUpscaleOrVaryParams(SharedFields):
    """
    Upscale model.
    """
    uov_method: UpscaleOrVaryMethod = "Upscale (2x)"


class ImgInpaintOrOutpaintParams(SharedFields):
    """
    Inpaint model.
    """
    outpaint_selections: str = Field("", description="A comma-separated string of 'Left', 'Right', 'Top', 'Bottom'")
    aspect_ratios_selection: str = "1152×896"
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

