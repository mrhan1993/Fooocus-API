"""Common models"""
from typing import List
from enum import Enum
from pydantic import (
    ConfigDict,
    BaseModel,
    TypeAdapter,
    Field
)

from fooocusapi.parameters import (
    default_sampler,
    default_scheduler,
    default_prompt_negative,
    default_aspect_ratio,
    default_base_model_name,
    default_refiner_model_name,
    default_refiner_switch,
    default_cfg_scale,
    default_styles,
    default_loras
)


class PerfomanceSelection(str, Enum):
    """Performance selection"""
    speed = 'Speed'
    quality = 'Quality'
    extreme_speed = 'Extreme Speed'


class Lora(BaseModel):
    """Common params lora model"""
    model_name: str
    weight: float = Field(default=0.5, ge=-2, le=2)

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )

LoraList = TypeAdapter(List[Lora])
default_loras_model = []
for lora in default_loras:
    if lora[0] != 'None':
        default_loras_model.append(
            Lora(model_name=lora[0], weight=lora[1])
        )
default_loras_json = LoraList.dump_json(default_loras_model)


class AdvancedParams(BaseModel):
    """Common params object AdvancedParams"""
    disable_preview: bool = Field(default=False,
                                  description="Disable preview during generation")
    adm_scaler_positive: float = Field(default=1.5,
                                       description="Positive ADM Guidance Scaler",
                                       ge=0.1, le=3.0)
    adm_scaler_negative: float = Field(default=0.8,
                                       description="Negative ADM Guidance Scaler",
                                       ge=0.1, le=3.0)
    adm_scaler_end: float = Field(default=0.3,
                                  description="ADM Guidance End At Step",
                                  ge=0.0, le=1.0)
    refiner_swap_method: str = Field(default='joint',
                                     description="Refiner swap method")
    adaptive_cfg: float = Field(default=7.0,
                                description="CFG Mimicking from TSNR",
                                ge=1.0, le=30.0)
    sampler_name: str = Field(default_sampler, description="Sampler")
    scheduler_name: str = Field(default_scheduler, description="Scheduler")
    overwrite_step: int = Field(default=-1,
                                description="Forced Overwrite of Sampling Step",
                                ge=-1, le=200)
    overwrite_switch: int = Field(default=-1,
                                  description="Forced Overwrite of Refiner Switch Step",
                                  ge=-1, le=200)
    overwrite_width: int = Field(default_factory=-1,
                                 description="Forced Overwrite of Generating Width",
                                 ge=-1, le=2048)
    overwrite_height: int = Field(default=-1,
                                  description="Forced Overwrite of Generating Height",
                                  ge=-1, le=2048)
    overwrite_vary_strength: float = Field(default=-1,
                                           description='Forced Overwrite of Denoising Strength of "Vary"',
                                           ge=-1, le=1.0)
    overwrite_upscale_strength: float = Field(default=-1,
                                              description='Forced Overwrite of Denoising Strength of "Upscale"',
                                              ge=-1, le=1.0)
    mixing_image_prompt_and_vary_upscale: bool = Field(default=False,
                                                       description="Mixing Image Prompt and Vary/Upscale")
    mixing_image_prompt_and_inpaint: bool = Field(default=False,
                                                   description="Mixing Image Prompt and Inpaint")
    debugging_cn_preprocessor: bool = Field(default=False, description="Debug Preprocessors")
    skipping_cn_preprocessor: bool = Field(default=False, description="Skip Preprocessors")
    controlnet_softness: float = Field(default=0.25, description="Softness of ControlNet", ge=0.0, le=1.0)
    canny_low_threshold: int = Field(default=64, description="Canny Low Threshold", ge=1, le=255)
    canny_high_threshold: int = Field(default=128, description="Canny High Threshold", ge=1, le=255)
    freeu_enabled: bool = Field(default=False, description="FreeU enabled")
    freeu_b1: float = Field(default=1.01, description="FreeU B1")
    freeu_b2: float = Field(default=1.02, description="FreeU B2")
    freeu_s1: float = Field(default=0.99, description="FreeU B3")
    freeu_s2: float = Field(default=0.95, description="FreeU B4")
    debugging_inpaint_preprocessor: bool = Field(default=False, description="Debug Inpaint Preprocessing")
    inpaint_disable_initial_latent: bool = Field(default=False, description="Disable initial latent in inpaint")
    inpaint_engine: str = Field(default='v2.6', description="Inpaint Engine")
    inpaint_strength: float = Field(default=1.0, description="Inpaint Denoising Strength", ge=0.0, le=1.0)
    inpaint_respective_field: float = Field(default=1.0, description="Inpaint Respective Field", ge=0.0, le=1.0)
    invert_mask_checkbox: bool = Field(default=False, description="Invert Mask")
    inpaint_erode_or_dilate: int = Field(default=0, description="Mask Erode or Dilate", ge=-64, le=64)



class CommonRequest(BaseModel):
    """All generate request based on this model"""
    prompt: str = ''
    negative_prompt: str = default_prompt_negative
    style_selections: List[str] = default_styles
    performance_selection: PerfomanceSelection = PerfomanceSelection.speed
    aspect_ratios_selection: str = default_aspect_ratio
    image_number: int = Field(default=1, description="Image number", ge=1, le=32)
    image_seed: int = Field(default=-1, description="Seed to generate image, -1 for random")
    sharpness: float = Field(default=2.0, ge=0.0, le=30.0)
    guidance_scale: float = Field(default=default_cfg_scale, ge=1.0, le=30.0)
    base_model_name: str = default_base_model_name
    refiner_model_name: str = default_refiner_model_name
    refiner_switch: float = Field(default=default_refiner_switch,
                                  description="Refiner Switch At",
                                  ge=0.1, le=1.0)
    loras: List[Lora] = Field(default=default_loras_model)
    advanced_params: AdvancedParams | None = AdvancedParams()
    require_base64: bool = Field(default=False, description="Return base64 data of generated image")
    async_process: bool = Field(default=False,
                                description="Specify whether the task is an asynchronous task")
    webhook_url: str | None = Field(default='',
                                    description="Optional URL for a webhook callback. If provided, "
                                    "a POST request will be send to this URL upon task completion or failure."
                                    " This allows for asynchronous notification of task status.")
