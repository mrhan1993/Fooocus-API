"""Common model for requests"""
from typing import List
from pydantic import (
    BaseModel,
    Field,
    ValidationError
)

from modules.config import (
    default_sampler,
    default_scheduler,
    default_prompt,
    default_prompt_negative,
    default_aspect_ratio,
    default_base_model_name,
    default_refiner_model_name,
    default_refiner_switch,
    default_cfg_scale,
    default_styles,
    default_overwrite_step,
    default_inpaint_engine_version,
    default_overwrite_switch,
    default_cfg_tsnr,
    default_sample_sharpness,
    default_vae,
    default_clip_skip
)

from modules.flags import clip_skip_max

from fooocusapi.models.common.base import (
    PerformanceSelection,
    Lora,
    default_loras_model
)

default_aspect_ratio = default_aspect_ratio.split(" ")[0].replace("×", "*")


class QueryJobRequest(BaseModel):
    """Query job request"""
    job_id: str = Field(description="Job ID to query")
    require_step_preview: bool = Field(
        default=False,
        description="Set to true will return preview image of generation steps at current time")


class AdvancedParams(BaseModel):
    """Common params object AdvancedParams"""
    disable_preview: bool = Field(False, description="Disable preview during generation")
    disable_intermediate_results: bool = Field(False, description="Disable intermediate results")
    disable_seed_increment: bool = Field(False, description="Disable Seed Increment")
    adm_scaler_positive: float = Field(1.5, description="Positive ADM Guidance Scaler", ge=0.1, le=3.0)
    adm_scaler_negative: float = Field(0.8, description="Negative ADM Guidance Scaler", ge=0.1, le=3.0)
    adm_scaler_end: float = Field(0.3, description="ADM Guidance End At Step", ge=0.0, le=1.0)
    adaptive_cfg: float = Field(default_cfg_tsnr, description="CFG Mimicking from TSNR", ge=1.0, le=30.0)
    clip_skip: int = Field(default_clip_skip, description="Clip Skip", ge=1, le=clip_skip_max)
    sampler_name: str = Field(default_sampler, description="Sampler")
    scheduler_name: str = Field(default_scheduler, description="Scheduler")
    overwrite_step: int = Field(default_overwrite_step, description="Forced Overwrite of Sampling Step", ge=-1, le=200)
    overwrite_switch: float = Field(default_overwrite_switch, description="Forced Overwrite of Refiner Switch Step", ge=-1, le=1)
    overwrite_width: int = Field(-1, description="Forced Overwrite of Generating Width", ge=-1, le=2048)
    overwrite_height: int = Field(-1, description="Forced Overwrite of Generating Height", ge=-1, le=2048)
    overwrite_vary_strength: float = Field(-1, description='Forced Overwrite of Denoising Strength of "Vary"', ge=-1, le=1.0)
    overwrite_upscale_strength: float = Field(-1, description='Forced Overwrite of Denoising Strength of "Upscale"', ge=-1, le=1.0)
    mixing_image_prompt_and_vary_upscale: bool = Field(False, description="Mixing Image Prompt and Vary/Upscale")
    mixing_image_prompt_and_inpaint: bool = Field(False, description="Mixing Image Prompt and Inpaint")
    debugging_cn_preprocessor: bool = Field(False, description="Debug Preprocessors")
    skipping_cn_preprocessor: bool = Field(False, description="Skip Preprocessors")
    canny_low_threshold: int = Field(64, description="Canny Low Threshold", ge=1, le=255)
    canny_high_threshold: int = Field(128, description="Canny High Threshold", ge=1, le=255)
    refiner_swap_method: str = Field('joint', description="Refiner swap method")
    controlnet_softness: float = Field(0.25, description="Softness of ControlNet", ge=0.0, le=1.0)
    freeu_enabled: bool = Field(False, description="FreeU enabled")
    freeu_b1: float = Field(1.01, description="FreeU B1")
    freeu_b2: float = Field(1.02, description="FreeU B2")
    freeu_s1: float = Field(0.99, description="FreeU B3")
    freeu_s2: float = Field(0.95, description="FreeU B4")
    debugging_inpaint_preprocessor: bool = Field(False, description="Debug Inpaint Preprocessing")
    inpaint_disable_initial_latent: bool = Field(False, description="Disable initial latent in inpaint")
    inpaint_engine: str = Field(default_inpaint_engine_version, description="Inpaint Engine")
    inpaint_strength: float = Field(1.0, description="Inpaint Denoising Strength", ge=0.0, le=1.0)
    inpaint_respective_field: float = Field(1.0, description="Inpaint Respective Field", ge=0.0, le=1.0)
    inpaint_mask_upload_checkbox: bool = Field(False, description="Upload Mask")
    invert_mask_checkbox: bool = Field(False, description="Invert Mask")
    inpaint_erode_or_dilate: int = Field(0, description="Mask Erode or Dilate", ge=-64, le=64)
    black_out_nsfw: bool = Field(False, description="Block out NSFW")
    vae_name: str = Field(default_vae, description="VAE name")


class CommonRequest(BaseModel):
    """All generate request based on this model"""
    prompt: str = default_prompt
    negative_prompt: str = default_prompt_negative
    style_selections: List[str] = default_styles
    performance_selection: PerformanceSelection = PerformanceSelection.speed
    aspect_ratios_selection: str = default_aspect_ratio
    image_number: int = Field(default=1, description="Image number", ge=1, le=32)
    image_seed: int = Field(default=-1, description="Seed to generate image, -1 for random")
    sharpness: float = Field(default=default_sample_sharpness, ge=0.0, le=30.0)
    guidance_scale: float = Field(default=default_cfg_scale, ge=1.0, le=30.0)
    base_model_name: str = default_base_model_name
    refiner_model_name: str = default_refiner_model_name
    refiner_switch: float = Field(default=default_refiner_switch, description="Refiner Switch At", ge=0.1, le=1.0)
    loras: List[Lora] = Field(default=default_loras_model)
    advanced_params: AdvancedParams = AdvancedParams()
    save_meta: bool = Field(default=True, description="Save meta data")
    meta_scheme: str = Field(default='fooocus', description="Meta data scheme, one of [fooocus, a111]")
    save_extension: str = Field(default='png', description="Save extension, one of [png, jpg, webp]")
    save_name: str = Field(default='', description="Image name for output image, default is job id + seq")
    read_wildcards_in_order: bool = Field(default=False, description="Read wildcards in order")
    require_base64: bool = Field(default=False, description="Return base64 data of generated image")
    async_process: bool = Field(default=False, description="Set to true will run async and return job info for retrieve generation result later")
    webhook_url: str | None = Field(default='', description="Optional URL for a webhook callback. If provided, the system will send a POST request to this URL upon task completion or failure."
                                                            " This allows for asynchronous notification of task status.")


def advanced_params_parser(advanced_params: str | None) -> AdvancedParams:
    """
    Parse advanced params, Convert to AdvancedParams
    Args:
        advanced_params: str, json format
    Returns:
        AdvancedParams object, if validate error return default value
    """
    if advanced_params is not None and len(advanced_params) > 0:
        try:
            advanced_params_obj = AdvancedParams.__pydantic_validator__.validate_json(advanced_params)
            return AdvancedParams(**advanced_params_obj)
        except ValidationError:
            return AdvancedParams()
    return AdvancedParams()
