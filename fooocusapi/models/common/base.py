"""Common models"""
from typing import List, Tuple
from enum import Enum
from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import (
    ValidationError,
    ConfigDict,
    BaseModel,
    TypeAdapter,
    Field
)
from pydantic_core import InitErrorDetails

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


class PerformanceSelection(str, Enum):
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


class UpscaleOrVaryMethod(str, Enum):
    """Upscale or Vary method"""
    subtle_variation = 'Vary (Subtle)'
    strong_variation = 'Vary (Strong)'
    upscale_15 = 'Upscale (1.5x)'
    upscale_2 = 'Upscale (2x)'
    upscale_fast = 'Upscale (Fast 2x)'
    upscale_custom = 'Upscale (Custom)'


class OutpaintExpansion(str, Enum):
    """Outpaint expansion"""
    left = 'Left'
    right = 'Right'
    top = 'Top'
    bottom = 'Bottom'


class ControlNetType(str, Enum):
    """ControlNet Type"""
    cn_ip = "ImagePrompt"
    cn_ip_face = "FaceSwap"
    cn_canny = "PyraCanny"
    cn_cpds = "CPDS"


class ImagePrompt(BaseModel):
    """Common params object ImagePrompt"""
    cn_img: UploadFile | None = Field(default=None)
    cn_stop: float | None = Field(default=None, ge=0, le=1)
    cn_weight: float | None = Field(default=None, ge=0, le=2, description="None for default value")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip)


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
    overwrite_width: int = Field(default=-1,
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
    controlnet_softness: float = Field(default=0.25,
                                       description="Softness of ControlNet",
                                       ge=0.0, le=1.0)
    canny_low_threshold: int = Field(default=64, description="Canny Low Threshold", ge=1, le=255)
    canny_high_threshold: int = Field(default=128, description="Canny High Threshold", ge=1, le=255)
    freeu_enabled: bool = Field(default=False, description="FreeU enabled")
    freeu_b1: float = Field(default=1.01, description="FreeU B1")
    freeu_b2: float = Field(default=1.02, description="FreeU B2")
    freeu_s1: float = Field(default=0.99, description="FreeU B3")
    freeu_s2: float = Field(default=0.95, description="FreeU B4")
    debugging_inpaint_preprocessor: bool = Field(default=False,
                                                 description="Debug Inpaint Preprocessing")
    inpaint_disable_initial_latent: bool = Field(default=False,
                                                 description="Disable initial latent in inpaint")
    inpaint_engine: str = Field(default='v2.6', description="Inpaint Engine")
    inpaint_strength: float = Field(default=1.0,
                                    description="Inpaint Denoising Strength",
                                    ge=0.0, le=1.0)
    inpaint_respective_field: float = Field(default=1.0,
                                            description="Inpaint Respective Field",
                                            ge=0.0, le=1.0)
    invert_mask_checkbox: bool = Field(default=False, description="Invert Mask")
    inpaint_erode_or_dilate: int = Field(default=0,
                                         description="Mask Erode or Dilate",
                                         ge=-64, le=64)


class CommonRequest(BaseModel):
    """All generate request based on this model"""
    prompt: str = ''
    negative_prompt: str = default_prompt_negative
    style_selections: List[str] = default_styles
    performance_selection: PerformanceSelection = PerformanceSelection.speed
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


def style_selection_parser(style_selections: str) -> List[str]:
    """
    Parse style selections, Convert to list
    Args:
        style_selections: str, comma separated Fooocus style selections
        e.g. Fooocus V2, Fooocus Enhance, Fooocus Sharp
    Returns:
        List[str]
    """
    style_selection_arr: List[str] = []
    if style_selections is None or len(style_selections) == 0:
        return []
    for part in style_selections:
        if len(part) > 0:
            for s in part.split(','):
                style = s.strip()
                style_selection_arr.append(style)
    return style_selection_arr


# todo: document not correct
def lora_parser(loras: str) -> List[Lora]:
    """
    Parse lora config, Convert to list
    Args:
        loras: str, comma separated model_name weight
        e.g. model_name1,weight1,model_name2,weight2
    Returns:
        List[Lora]
    """
    loras_model: List[Lora] = []
    if loras is None or len(loras) == 0:
        return loras_model
    try:
        loras_model = LoraList.validate_json(loras)
        return loras_model
    except ValidationError as ve:
        errs = ve.errors()
        raise RequestValidationError from errs


def advanced_params_parser(advanced_params: str | None) -> AdvancedParams | None:
    """
    Parse advanced params, Convert to AdvancedParams
    Args:
        advanced_params: str, json format
    Returns:
        AdvancedParams
    """
    advanced_params_obj = None
    if advanced_params is not None and len(advanced_params) > 0:
        try:
            advanced_params_obj = AdvancedParams.__pydantic_validator__.validate_json(
                advanced_params)
            return advanced_params_obj
        except ValidationError as ve:
            errs = ve.errors()
            raise RequestValidationError from errs
    return advanced_params_obj


def outpaint_selections_parser(outpaint_selections: str) -> List[OutpaintExpansion]:
    """
    Parse outpaint selections, Convert to list
    Args:
        outpaint_selections: str, comma separated Left, Right, Top, Bottom
        e.g. Left, Right, Top, Bottom
    Returns:
        List[OutpaintExpansion]
    """
    outpaint_selections_arr: List[OutpaintExpansion] = []
    if outpaint_selections is None or len(outpaint_selections) == 0:
        return []
    for part in outpaint_selections:
        if len(part) > 0:
            for s in part.split(','):
                try:
                    expansion = OutpaintExpansion(s)
                    outpaint_selections_arr.append(expansion)
                except ValueError:
                    errs = InitErrorDetails(
                        type='enum', loc=['outpaint_selections'],
                        input=outpaint_selections,
                        ctx={
                            'expected': "str, comma separated Left, Right, Top, Bottom"
                        })
                    raise RequestValidationError from errs
    return outpaint_selections_arr


# todo: document not correct
def image_prompt_parser(image_prompts_config: List[Tuple]) -> List[ImagePrompt]:
    """
    Image prompt parser, Convert to List[ImagePrompt]
    Args:
        image_prompts_config: List[Tuple]
        e.g. ('image1.jpg', 0.5, 1.0, 'normal'), ('image2.jpg', 0.5, 1.0, 'normal')
    returns:
        List[ImagePrompt]
    """
    image_prompts: List[ImagePrompt] = []
    if image_prompts_config is None or len(image_prompts_config) == 0:
        return []
    for config in image_prompts_config:
        cn_img, cn_stop, cn_weight, cn_type = config
        image_prompts.append(ImagePrompt(cn_img=cn_img, cn_stop=cn_stop,
                                         cn_weight=cn_weight, cn_type=cn_type))
    return image_prompts
