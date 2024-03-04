"""V1 API models"""
from typing import List, Tuple
from enum import Enum

from fastapi import Form, UploadFile
from fastapi.params import File
from fastapi.exceptions import RequestValidationError

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_core import InitErrorDetails

from fooocusapi.parameters import (GenerationFinishReason,
                                   default_styles,
                                   default_base_model_name,
                                   default_refiner_model_name,
                                   default_refiner_switch,
                                   default_cfg_scale,
                                   default_prompt_negative,
                                   default_aspect_ratio)

from fooocusapi.task_queue import TaskType


class UpscaleOrVaryMethod(str, Enum):
    subtle_variation = 'Vary (Subtle)'
    strong_variation = 'Vary (Strong)'
    upscale_15 = 'Upscale (1.5x)'
    upscale_2 = 'Upscale (2x)'
    upscale_fast = 'Upscale (Fast 2x)'
    upscale_custom = 'Upscale (Custom)'

class OutpaintExpansion(str, Enum):
    left = 'Left'
    right = 'Right'
    top = 'Top'
    bottom = 'Bottom'


class ControlNetType(str, Enum):
    cn_ip = "ImagePrompt"
    cn_ip_face = "FaceSwap"
    cn_canny = "PyraCanny"
    cn_cpds = "CPDS"


class ImagePrompt(BaseModel):
    cn_img: UploadFile | None = Field(default=None)
    cn_stop: float | None = Field(default=None, ge=0, le=1)
    cn_weight: float | None = Field(default=None, ge=0, le=2, description="None for default value")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip)





def style_selection_parser(style_selections: str) -> List[str]:
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
    loras_model: List[Lora] = []
    if loras is None or len(loras) == 0:
        return []
    try:
        loras_model = LoraList.validate_json(loras)
        return loras_model
    except ValidationError as ve:
        errs = ve.errors()
        raise RequestValidationError(errors=[errs])

def advanced_params_parser(advanced_params: str | None) -> AdvancedParams:
    advanced_params_obj = None
    if advanced_params is not None and len(advanced_params) > 0:
        try:
            advanced_params_obj = AdvancedParams.__pydantic_validator__.validate_json(advanced_params)
            return advanced_params_obj
        except ValidationError as ve:
            errs = ve.errors()
            raise RequestValidationError(errors=[errs])
    return advanced_params_obj

def oupaint_selections_parser(outpaint_selections: str) -> List[OutpaintExpansion]:
    outpaint_selections_arr: List[OutpaintExpansion] = []
    if outpaint_selections is None or len(outpaint_selections) == 0:
        return []
    for part in outpaint_selections:
        if len(part) > 0:
            for s in part.split(','):
                try:
                    expansion = OutpaintExpansion(s)
                    outpaint_selections_arr.append(expansion)
                except ValueError as ve:
                    err = InitErrorDetails(type='enum', loc=['outpaint_selections'],
                                           input=outpaint_selections,
                                           ctx={
                                               'expected': "Literal 'Left', 'Right', 'Top', 'Bottom' seperated by comma"
                                            })
                    raise RequestValidationError(errors=[err])
    return outpaint_selections_arr

def image_prompt_parser(image_prompts_config: List[Tuple]) -> List[ImagePrompt]:
    image_prompts: List[ImagePrompt] = []
    if image_prompts_config is None or len(image_prompts_config) == 0:
        return []
    for config in image_prompts_config:
        cn_img, cn_stop, cn_weight, cn_type = config
        image_prompts.append(ImagePrompt(cn_img=cn_img, cn_stop=cn_stop,
                                         cn_weight=cn_weight, cn_type=cn_type))
    return image_prompts


class ImgUpscaleOrVaryRequest(Text2ImgRequest):
    input_image: UploadFile
    uov_method: UpscaleOrVaryMethod
    upscale_value: float | None

    @classmethod
    def as_form(cls, input_image: UploadFile = Form(description="Init image for upsacale or outpaint"),
                uov_method: UpscaleOrVaryMethod = Form(),
                upscale_value: float | None = Form(None, description="Upscale custom value, None for default value", ge=1.0, le=5.0),
                prompt: str = Form(''),
                negative_prompt: str = Form(default_prompt_negative),
                style_selections: List[str] = Form(default_styles, description="Fooocus style selections, seperated by comma"),
                performance_selection: PerfomanceSelection = Form(PerfomanceSelection.speed, description="Performance Selection, one of 'Speed','Quality','Extreme Speed'"),
                aspect_ratios_selection: str = Form(default_aspect_ratio, description="Aspect Ratios Selection, default 1152*896"),
                image_number: int = Form(default=1, description="Image number", ge=1, le=32),
                image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
                sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
                guidance_scale: float = Form(default=default_cfg_scale, ge=1.0, le=30.0),
                base_model_name: str = Form(default_base_model_name, description="checkpoint file name"),
                refiner_model_name: str = Form(default_refiner_model_name, description="refiner file name"),
                refiner_switch: float = Form(default=default_refiner_switch, description="Refiner Switch At", ge=0.1, le=1.0),
                loras: str | None = Form(default=default_loras_json, description='Lora config in JSON. Format as [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}]'),
                advanced_params: str | None = Form(default=None, description="Advanced parameters in JSON"),
                require_base64: bool = Form(default=False, description="Return base64 data of generated image"),
                async_process: bool = Form(default=False, description="Set to true will run async and return job info for retrieve generataion result later"),
                ):
        style_selection_arr = style_selection_parser(style_selections)
        loras_model = lora_parser(loras)
        advanced_params_obj = advanced_params_parser(advanced_params)

        return cls(input_image=input_image, uov_method=uov_method,upscale_value=upscale_value,
                   prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name, refiner_switch=refiner_switch,
                   loras=loras_model, advanced_params=advanced_params_obj, require_base64=require_base64, async_process=async_process)


class ImgInpaintOrOutpaintRequest(Text2ImgRequest):
    input_image: UploadFile | None
    input_mask: UploadFile | None
    inpaint_additional_prompt: str | None
    outpaint_selections: List[OutpaintExpansion]
    outpaint_distance_left: int
    outpaint_distance_right: int
    outpaint_distance_top: int
    outpaint_distance_bottom: int

    @classmethod
    def as_form(cls, input_image: UploadFile = Form(description="Init image for inpaint or outpaint"),
                input_mask: UploadFile = Form(File(None), description="Inpaint or outpaint mask"),
                inpaint_additional_prompt: str | None = Form(None, description="Describe what you want to inpaint"),
                outpaint_selections: List[str] = Form([], description="Outpaint expansion selections, literal 'Left', 'Right', 'Top', 'Bottom' seperated by comma"),
                outpaint_distance_left: int = Form(default=0, description="Set outpaint left distance, -1 for default"),
                outpaint_distance_right: int = Form(default=0, description="Set outpaint right distance, -1 for default"),
                outpaint_distance_top: int = Form(default=0, description="Set outpaint top distance, -1 for default"),
                outpaint_distance_bottom: int = Form(default=0, description="Set outpaint bottom distance, -1 for default"),
                prompt: str = Form(''),
                negative_prompt: str = Form(default_prompt_negative),
                style_selections: List[str] = Form(default_styles, description="Fooocus style selections, seperated by comma"),
                performance_selection: PerfomanceSelection = Form(PerfomanceSelection.speed, description="Performance Selection, one of 'Speed','Quality','Extreme Speed'"),
                aspect_ratios_selection: str = Form(default_aspect_ratio, description="Aspect Ratios Selection, default 1152*896"),
                image_number: int = Form(default=1, description="Image number", ge=1, le=32),
                image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
                sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
                guidance_scale: float = Form(default=default_cfg_scale, ge=1.0, le=30.0),
                base_model_name: str = Form(default_base_model_name),
                refiner_model_name: str = Form(default_refiner_model_name),
                refiner_switch: float = Form(default=default_refiner_switch, description="Refiner Switch At", ge=0.1, le=1.0),
                loras: str | None = Form(default=default_loras_json, description='Lora config in JSON. Format as [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}]'),
                advanced_params: str| None = Form(default=None, description="Advanced parameters in JSON"),
                require_base64: bool = Form(default=False, description="Return base64 data of generated image"),
                async_process: bool = Form(default=False, description="Set to true will run async and return job info for retrieve generataion result later"),
                ):

        if isinstance(input_mask, File):
            input_mask = None
        
        outpaint_selections_arr = oupaint_selections_parser(outpaint_selections)
        style_selection_arr = style_selection_parser(style_selections)
        loras_model = lora_parser(loras)
        advanced_params_obj = advanced_params_parser(advanced_params)

        return cls(input_image=input_image, input_mask=input_mask, inpaint_additional_prompt=inpaint_additional_prompt,
                   outpaint_selections=outpaint_selections_arr,outpaint_distance_left=outpaint_distance_left,
                   outpaint_distance_right=outpaint_distance_right, outpaint_distance_top=outpaint_distance_top,
                   outpaint_distance_bottom=outpaint_distance_bottom, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name, refiner_switch=refiner_switch,
                   loras=loras_model, advanced_params=advanced_params_obj, require_base64=require_base64, async_process=async_process)


class ImgPromptRequest(ImgInpaintOrOutpaintRequest):
    image_prompts: List[ImagePrompt]

    @classmethod
    def as_form(cls, input_image: UploadFile = Form(File(None), description="Init image for inpaint or outpaint"),
                input_mask: UploadFile = Form(File(None), description="Inpaint or outpaint mask"),
                inpaint_additional_prompt: str | None = Form(None, description="Describe what you want to inpaint"),
                outpaint_selections: List[str] = Form([], description="Outpaint expansion selections, literal 'Left', 'Right', 'Top', 'Bottom' seperated by comma"),
                outpaint_distance_left: int = Form(default=0, description="Set outpaint left distance, 0 for default"),
                outpaint_distance_right: int = Form(default=0, description="Set outpaint right distance, 0 for default"),
                outpaint_distance_top: int = Form(default=0, description="Set outpaint top distance, 0 for default"),
                outpaint_distance_bottom: int = Form(default=0, description="Set outpaint bottom distance, 0 for default"),
                cn_img1: UploadFile = Form(File(None), description="Input image for image prompt"),
                cn_stop1: float | None = Form(
                    default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
                cn_weight1: float | None = Form(
                    default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
                cn_type1: ControlNetType = Form(
                    default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
                cn_img2: UploadFile = Form(
                    File(None), description="Input image for image prompt"),
                cn_stop2: float | None = Form(
                    default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
                cn_weight2: float | None = Form(
                    default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
                cn_type2: ControlNetType = Form(
                    default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
                cn_img3: UploadFile = Form(
                    File(None), description="Input image for image prompt"),
                cn_stop3: float | None = Form(
                    default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
                cn_weight3: float | None = Form(
                    default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
                cn_type3: ControlNetType = Form(
                    default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
                cn_img4: UploadFile = Form(
                    File(None), description="Input image for image prompt"),
                cn_stop4: float | None = Form(
                    default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
                cn_weight4: float | None = Form(
                    default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
                cn_type4: ControlNetType = Form(
                    default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
                prompt: str = Form(''),
                negative_prompt: str = Form(default_prompt_negative),
                style_selections: List[str] = Form(default_styles, description="Fooocus style selections, seperated by comma"),
                performance_selection: PerfomanceSelection = Form(
                    PerfomanceSelection.speed),
                aspect_ratios_selection: str = Form(default_aspect_ratio),
                image_number: int = Form(
                    default=1, description="Image number", ge=1, le=32),
                image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
                sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
                guidance_scale: float = Form(default=default_cfg_scale, ge=1.0, le=30.0),
                base_model_name: str = Form(default_base_model_name),
                refiner_model_name: str = Form(default_refiner_model_name),
                refiner_switch: float = Form(default=default_refiner_switch, description="Refiner Switch At", ge=0.1, le=1.0),
                loras: str | None = Form(default=default_loras_json, description='Lora config in JSON. Format as [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}]'),
                advanced_params: str| None = Form(default=None, description="Advanced parameters in JSON"),
                require_base64: bool = Form(default=False, description="Return base64 data of generated image"),
                async_process: bool = Form(default=False, description="Set to true will run async and return job info for retrieve generataion result later"),
                ):
        if isinstance(input_image, File):
            input_image = None
        if isinstance(input_mask, File):
            input_mask = None
        if isinstance(cn_img1, File):
            cn_img1 = None
        if isinstance(cn_img2, File):
            cn_img2 = None
        if isinstance(cn_img3, File):
            cn_img3 = None
        if isinstance(cn_img4, File):
            cn_img4 = None

        outpaint_selections_arr = oupaint_selections_parser(outpaint_selections)

        image_prompt_config = [(cn_img1, cn_stop1, cn_weight1, cn_type1),
                               (cn_img2, cn_stop2, cn_weight2, cn_type2),
                               (cn_img3, cn_stop3, cn_weight3, cn_type3),
                               (cn_img4, cn_stop4, cn_weight4, cn_type4)]
        image_prompts = image_prompt_parser(image_prompt_config)
        style_selection_arr = style_selection_parser(style_selections)
        loras_model = lora_parser(loras)
        advanced_params_obj = advanced_params_parser(advanced_params)

        return cls(input_image=input_image, input_mask=input_mask, inpaint_additional_prompt=inpaint_additional_prompt, outpaint_selections=outpaint_selections_arr,
                   outpaint_distance_left=outpaint_distance_left, outpaint_distance_right=outpaint_distance_right, outpaint_distance_top=outpaint_distance_top, outpaint_distance_bottom=outpaint_distance_bottom,
                   image_prompts=image_prompts, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name, refiner_switch=refiner_switch,
                   loras=loras_model, advanced_params=advanced_params_obj, require_base64=require_base64, async_process=async_process)
