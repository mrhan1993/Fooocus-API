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

from fooocusapi.configs.default import default_loras


class PerformanceSelection(str, Enum):
    """Performance selection"""
    speed = 'Speed'
    quality = 'Quality'
    extreme_speed = 'Extreme Speed'
    lightning = 'Lightning'
    hyper_sd = 'Hyper-SD'


class Lora(BaseModel):
    """Common params lora model"""
    enabled: bool
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
            Lora(
                enabled=lora[0],
                model_name=lora[1],
                weight=lora[2])
        )
default_loras_json = LoraList.dump_json(default_loras_model)


class MaskModel(str, Enum):
    """Inpaint mask model"""
    u2net = "u2net"
    u2netp = "u2netp"
    u2net_human_seg = "u2net_human_seg"
    u2net_cloth_seg = "u2net_cloth_seg"
    silueta = "silueta"
    isnet_general_use = "isnet-general-use"
    isnet_anime = "isnet-anime"
    sam = "sam"


class EnhanceCtrlNets(BaseModel):
    enhance_enabled: bool = Field(default=False, description="Enable enhance control nets")
    enhance_mask_dino_prompt: str = Field(default="face", description="Mask dino prompt, this is necessary, error if no value. usual values: face, eye, mouth, hair, hand, body")
    enhance_prompt: str = Field(default="", description="Prompt")
    enhance_negative_prompt: str = Field(default="", description="Negative prompt")
    enhance_mask_model: MaskModel = Field(default=MaskModel.sam, description="Mask model")
    enhance_mask_cloth_category: str = Field(default="full", description="Mask cloth category")
    enhance_mask_sam_model: str = Field(default="vit_b", description="one of vit_b vit_h vit_l")
    enhance_mask_text_threshold: float = Field(default=0.25, ge=0, le=1, description="Mask text threshold")
    enhance_mask_box_threshold: float = Field(default=0.3, ge=0, le=1, description="Mask box threshold")
    enhance_mask_sam_max_detections: int = Field(default=0, ge=0, le=10, description="Mask sam max detections, Set to 0 to detect all")
    enhance_inpaint_disable_initial_latent: bool = Field(default=False, description="Inpaint disable initial latent")
    enhance_inpaint_engine: str = Field(default="v2.6", description="Inpaint engine")
    enhance_inpaint_strength: float = Field(default=1, ge=0, le=1, description="Inpaint strength")
    enhance_inpaint_respective_field: float = Field(default=0.618, ge=0, le=1, description="Inpaint respective field")
    enhance_inpaint_erode_or_dilate: float = Field(default=0, ge=-64, le=64, description="Inpaint erode or dilate")
    enhance_mask_invert: bool = Field(default=False, description="Inpaint mask invert")


class GenerateMaskRequest(BaseModel):
    """
    generate mask request
    """
    image: str = Field(description="Image url or base64")
    mask_model: MaskModel = Field(default=MaskModel.isnet_general_use, description="Mask model")
    cloth_category: str = Field(default="full", description="Mask cloth category")
    dino_prompt_text: str = Field(default="", description="Detection prompt, Use singular whenever possible")
    sam_model: str = Field(default="vit_b", description="one of vit_b vit_h vit_l")
    box_threshold: float = Field(default=0.3, ge=0, le=1, description="Mask box threshold")
    text_threshold: float = Field(default=0.25, ge=0, le=1, description="Mask text threshold")
    sam_max_detections: int = Field(default=0, ge=0, le=10, description="Mask sam max detections, Set to 0 to detect all")
    dino_erode_or_dilate: float = Field(default=0, ge=-64, le=64, description="Mask dino erode or dilate")
    dino_debug: bool = Field(default=False, description="Mask dino debug")


class UpscaleOrVaryMethod(str, Enum):
    """Upscale or Vary method"""
    disabled = 'Disabled'
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
    cn_stop: float | None = Field(default=0.5, ge=0, le=1)
    cn_weight: float | None = Field(default=0.6, ge=0, le=2, description="None for default value")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip)


class DescribeImageType(str, Enum):
    """Image type for image to prompt"""
    photo = 'Photo'
    anime = 'Anime'


class ImageMetaScheme(str, Enum):
    """Scheme for save image meta
    Attributes:
        Fooocus: json format
        A111: string
    """
    Fooocus = 'fooocus'
    A111 = 'a111'


def style_selection_parser(style_selections: str | List[str]) -> List[str]:
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


def lora_parser(loras: str) -> List[Lora]:
    """
    Parse lora config, Convert to list
    Args:
        loras: a json string for loras
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


def outpaint_selections_parser(outpaint_selections: str | list[str]) -> List[OutpaintExpansion]:
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
                        type='enum',
                        loc=tuple('outpaint_selections'),
                        input=outpaint_selections,
                        ctx={
                            'expected': "str, comma separated Left, Right, Top, Bottom"
                        })
                    raise RequestValidationError from errs
    return outpaint_selections_arr


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
        image_prompts.append(ImagePrompt(
            cn_img=cn_img,
            cn_stop=cn_stop,
            cn_weight=cn_weight,
            cn_type=cn_type))
    return image_prompts
