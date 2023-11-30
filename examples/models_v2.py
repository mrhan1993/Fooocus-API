from pydantic import BaseModel, Field
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
    subtle_variation = 'Vary (Subtle)'
    strong_variation = 'Vary (Strong)'
    upscale_15 = 'Upscale (1.5x)'
    upscale_2 = 'Upscale (2x)'
    upscale_fast = 'Upscale (Fast 2x)'

class ControlNetEnum(str, Enum):
    """
    ControlNet enum.
    """
    imagePrompt = "ImagePrompt"
    faceSwap = "FaceSwap"
    pyraCanny = "PyraCanny"
    cpds = "CPDS"

class ImagePrompt(BaseModel):
    cn_img: str | None = None
    cn_stop: float | None = 0.6
    cn_weight: float | None = 0.6
    cn_type: ControlNetEnum = "ImagePrompt"

class OutpaintExpansion(str, Enum):
    left = 'Left'
    right = 'Right'
    top = 'Top'
    bottom = 'Bottom'

class Text2ImgParams(BaseModel):
    """
    Text2ImgPrompt model.
    """
    prompt: str = ""
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

class ImgUpscaleOrVaryParams(Text2ImgParams):
    uov_method: UpscaleOrVaryMethod = "Upscale (2x)"
    input_image: str = Field(description="Init image for upsacale or outpaint as base64")

class ImgInpaintOrOutpaintParams(Text2ImgParams):
    input_image: str
    input_mask: str | None = None
    inpaint_additional_prompt: str | None = None
    outpaint_selections: List[OutpaintExpansion] = ["Left", "Right", "Top", "Bottom"]
    outpaint_distance_left: int = 0
    outpaint_distance_right: int = 0
    outpaint_distance_top: int = 0
    outpaint_distance_bottom: int = 0

class ImgPromptParams(Text2ImgParams):
    image_prompts: List[ImagePrompt] = [ImagePrompt()]
