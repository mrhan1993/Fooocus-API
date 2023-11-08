import json
from fastapi import Form, UploadFile
from fastapi.params import File
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, parse_obj_as
from typing import List
from enum import Enum

from pydantic_core import InitErrorDetails
from fooocusapi.parameters import GenerationFinishReason, defualt_styles, default_base_model_name, default_refiner_model_name, default_refiner_switch, default_lora_name, default_lora_weight, default_cfg_scale, default_prompt_negative, default_aspect_ratio, default_sampler, default_scheduler
from fooocusapi.task_queue import TaskType
import modules.flags as flags


class Lora(BaseModel):
    model_name: str
    weight: float = Field(default=0.5, ge=-2, le=2)

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )


class PerfomanceSelection(str, Enum):
    speed = 'Speed'
    quality = 'Quality'
    

class UpscaleOrVaryMethod(str, Enum):
    subtle_variation = 'Vary (Subtle)'
    strong_variation = 'Vary (Strong)'
    upscale_15 = 'Upscale (1.5x)'
    upscale_2 = 'Upscale (2x)'
    upscale_fast = 'Upscale (Fast 2x)'


class OutpaintExpansion(str, Enum):
    left = 'Left'
    right = 'Right'
    top = 'Top'
    bottom = 'Bottom'


class ControlNetType(str, Enum):
    cn_ip = 'Image Prompt'
    cn_canny = 'PyraCanny'
    cn_cpds = 'CPDS'


class ImagePrompt(BaseModel):
    cn_img: UploadFile | None = Field(default=None)
    cn_stop: float = Field(default=0.4, ge=0, le=1)
    cn_weight: float | None = Field(
        default=None, ge=0, le=2, description="None for default value")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip)


class AdvancedParams(BaseModel):
    adm_scaler_positive: float = Field(1.5, description="Positive ADM Guidance Scaler", ge=0.1, le=3.0)
    adm_scaler_negative: float = Field(0.8, description="Negative ADM Guidance Scaler", ge=0.1, le=3.0)
    adm_scaler_end: float = Field(0.3, description="ADM Guidance End At Step", ge=0.0, le=1.0)
    refiner_swap_method: str = Field('joint', description="Refiner swap method")
    adaptive_cfg: float = Field(7.0, description="CFG Mimicking from TSNR", ge=1.0, le=30.0)
    sampler_name: str = Field(default_sampler, description="Sampler")
    scheduler_name: str = Field(default_scheduler, description="Scheduler")
    overwrite_step: int = Field(-1, description="Forced Overwrite of Sampling Step", ge=-1, le=200)
    overwrite_switch: int = Field(-1, description="Forced Overwrite of Refiner Switch Step", ge=-1, le=200)
    overwrite_width: int = Field(-1, description="Forced Overwrite of Generating Width", ge=-1, le=2048)
    overwrite_height: int = Field(-1, description="Forced Overwrite of Generating Height", ge=-1, le=2048)
    overwrite_vary_strength: float = Field(-1, description='Forced Overwrite of Denoising Strength of "Vary"', ge=-1, le=1.0)
    overwrite_upscale_strength: float = Field(-1, description='Forced Overwrite of Denoising Strength of "Upscale"', ge=-1, le=1.0)
    mixing_image_prompt_and_vary_upscale: bool = Field(False, description="Mixing Image Prompt and Vary/Upscale")
    mixing_image_prompt_and_inpaint: bool = Field(False, description="Mixing Image Prompt and Inpaint")
    debugging_cn_preprocessor: bool = Field(False, description="Debug Preprocessors")
    controlnet_softness: float = Field(0.25, description="Softness of ControlNet", ge=0.0, le=1.0)
    canny_low_threshold: int = Field(64, description="Canny Low Threshold", ge=1, le=255)
    canny_high_threshold: int = Field(128, description="Canny High Threshold", ge=1, le=255)
    inpaint_engine: str = Field('v1', description="Inpaint Engine")
    freeu_enabled: bool = Field(False, description="FreeU enabled")
    freeu_b1: float = Field(1.01, description="FreeU B1")
    freeu_b2: float = Field(1.02, description="FreeU B2")
    freeu_s1: float = Field(0.99, description="FreeU B3")
    freeu_s2: float = Field(0.95, description="FreeU B4")


class Text2ImgRequest(BaseModel):
    prompt: str = ''
    negative_prompt: str = default_prompt_negative
    style_selections: List[str] = defualt_styles
    performance_selection: PerfomanceSelection = PerfomanceSelection.speed
    aspect_ratios_selection: str = default_aspect_ratio
    image_number: int = Field(
        default=1, description="Image number", ge=1, le=32)
    image_seed: int = Field(default=-1, description="Seed to generate image, -1 for random")
    sharpness: float = Field(default=2.0, ge=0.0, le=30.0)
    guidance_scale: float = Field(default=default_cfg_scale, ge=1.0, le=30.0)
    base_model_name: str = default_base_model_name
    refiner_model_name: str = default_refiner_model_name
    refiner_switch: float = Field(default=default_refiner_switch, description="Refiner Switch At", ge=0.1, le=1.0)
    loras: List[Lora] = Field(default=[
        Lora(model_name=default_lora_name, weight=default_lora_weight)])
    advanced_params: AdvancedParams | None = Field(deafult=None, description="Advanced parameters")
    require_base64: bool = Field(default=False, description="Return base64 data of generated image")
    async_process: bool = Field(default=False, description="Set to true will run async and return job info for retrieve generataion result later")


class ImgUpscaleOrVaryRequest(Text2ImgRequest):
    input_image: UploadFile
    uov_method: UpscaleOrVaryMethod

    @classmethod
    def as_form(cls, input_image: UploadFile = Form(description="Init image for upsacale or outpaint"),
                uov_method: UpscaleOrVaryMethod = Form(),
                prompt: str = Form(''),
                negative_prompt: str = Form(default_prompt_negative),
                style_selections: List[str] = Form(defualt_styles, description="Fooocus style selections, seperated by comma"),
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
                l1: str | None = Form(default_lora_name),
                w1: float = Form(default=default_lora_weight, ge=-2, le=2),
                l2: str | None = Form(None),
                w2: float = Form(default=default_lora_weight, ge=-2, le=2),
                l3: str | None = Form(None),
                w3: float = Form(default=default_lora_weight, ge=-2, le=2),
                l4: str | None = Form(None),
                w4: float = Form(default=default_lora_weight, ge=-2, le=2),
                l5: str | None = Form(None),
                w5: float = Form(default=default_lora_weight, ge=-2, le=2),
                advanced_params: str| None = Form(default=None, description="Advanced parameters in JSON"),
                require_base64: bool = Form(default=False, description="Return base64 data of generated image"),
                async_process: bool = Form(default=False, description="Set to true will run async and return job info for retrieve generataion result later"),
                ):
        style_selection_arr: List[str] = []
        for part in style_selections:
            if len(part) > 0:
                for s in part.split(','):
                    style = s.strip()
                    style_selection_arr.append(style)

        loras: List[Lora] = []
        lora_config = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]
        for config in lora_config:
            lora_model, lora_weight = config
            if lora_model is not None and len(lora_model) > 0:
                loras.append(Lora(model_name=lora_model, weight=lora_weight))

        advanced_params_obj = None
        if advanced_params != None and len(advanced_params) > 0:
            try:
                advanced_params_obj = AdvancedParams.__pydantic_validator__.validate_json(advanced_params)
            except ValidationError as ve:
                errs = ve.errors()
                raise RequestValidationError(errors=[errs])

        return cls(input_image=input_image, uov_method=uov_method, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name, refiner_switch=refiner_switch,
                   loras=loras, advanced_params=advanced_params_obj, require_base64=require_base64, async_process=async_process)


class ImgInpaintOrOutpaintRequest(Text2ImgRequest):
    input_image: UploadFile
    input_mask: UploadFile | None
    outpaint_selections: List[OutpaintExpansion]

    @classmethod
    def as_form(cls, input_image: UploadFile = Form(description="Init image for inpaint or outpaint"),
                input_mask: UploadFile = Form(
                    File(None), description="Inpaint or outpaint mask"),
                outpaint_selections: List[str] = Form(
                    [], description="Outpaint expansion selections, literal 'Left', 'Right', 'Top', 'Bottom' seperated by comma"),
                prompt: str = Form(''),
                negative_prompt: str = Form(default_prompt_negative),
                style_selections: List[str] = Form(defualt_styles, description="Fooocus style selections, seperated by comma"),
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
                l1: str | None = Form(default_lora_name),
                w1: float = Form(default=default_lora_weight, ge=-2, le=2),
                l2: str | None = Form(None),
                w2: float = Form(default=default_lora_weight, ge=-2, le=2),
                l3: str | None = Form(None),
                w3: float = Form(default=default_lora_weight, ge=-2, le=2),
                l4: str | None = Form(None),
                w4: float = Form(default=default_lora_weight, ge=-2, le=2),
                l5: str | None = Form(None),
                w5: float = Form(default=default_lora_weight, ge=-2, le=2),
                advanced_params: str| None = Form(default=None, description="Advanced parameters in JSON"),
                require_base64: bool = Form(default=False, description="Return base64 data of generated image"),
                async_process: bool = Form(default=False, description="Set to true will run async and return job info for retrieve generataion result later"),
                ):

        if isinstance(input_mask, File):
            input_mask = None
        
        outpaint_selections_arr: List[OutpaintExpansion] = []
        for part in outpaint_selections:
            if len(part) > 0:
                for s in part.split(','):
                    try:
                        expansion = OutpaintExpansion(s)
                        outpaint_selections_arr.append(expansion)
                    except ValueError as ve:
                        err = InitErrorDetails(type='enum', loc=['outpaint_selections'], input=outpaint_selections, ctx={
                            'expected': "Literal 'Left', 'Right', 'Top', 'Bottom' seperated by comma"})
                        raise RequestValidationError(errors=[err])

        style_selection_arr: List[str] = []
        for part in style_selections:
            if len(part) > 0:
                for s in part.split(','):
                    style = s.strip()
                    style_selection_arr.append(style)

        loras: List[Lora] = []
        lora_config = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]
        for config in lora_config:
            lora_model, lora_weight = config
            if lora_model is not None and len(lora_model) > 0:
                loras.append(Lora(model_name=lora_model, weight=lora_weight))

        advanced_params_obj = None
        if advanced_params != None and len(advanced_params) > 0:
            try:
                advanced_params_obj = AdvancedParams.__pydantic_validator__.validate_json(advanced_params)
            except ValidationError as ve:
                errs = ve.errors()
                raise RequestValidationError(errors=[errs])

        return cls(input_image=input_image, input_mask=input_mask, outpaint_selections=outpaint_selections_arr, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name, refiner_switch=refiner_switch,
                   loras=loras, advanced_params=advanced_params_obj, require_base64=require_base64, async_process=async_process)


class ImgPromptRequest(Text2ImgRequest):
    image_prompts: List[ImagePrompt]

    @classmethod
    def as_form(cls, cn_img1: UploadFile = Form(File(None), description="Input image for image prompt"),
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
                style_selections: List[str] = Form(defualt_styles, description="Fooocus style selections, seperated by comma"),
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
                l1: str | None = Form(default_lora_name),
                w1: float = Form(default=default_lora_weight, ge=-2, le=2),
                l2: str | None = Form(None),
                w2: float = Form(default=default_lora_weight, ge=-2, le=2),
                l3: str | None = Form(None),
                w3: float = Form(default=default_lora_weight, ge=-2, le=2),
                l4: str | None = Form(None),
                w4: float = Form(default=default_lora_weight, ge=-2, le=2),
                l5: str | None = Form(None),
                w5: float = Form(default=default_lora_weight, ge=-2, le=2),
                advanced_params: str| None = Form(default=None, description="Advanced parameters in JSON"),
                require_base64: bool = Form(default=False, description="Return base64 data of generated image"),
                async_process: bool = Form(default=False, description="Set to true will run async and return job info for retrieve generataion result later"),
                ):
        if isinstance(cn_img1, File):
            cn_img1 = None
        if isinstance(cn_img2, File):
            cn_img2 = None
        if isinstance(cn_img3, File):
            cn_img3 = None
        if isinstance(cn_img4, File):
            cn_img4 = None

        image_prompts: List[ImagePrompt] = []
        image_prompt_config = [(cn_img1, cn_stop1, cn_weight1, cn_type1), (cn_img2, cn_stop2, cn_weight2, cn_type2),
                               (cn_img3, cn_stop3, cn_weight3, cn_type3), (cn_img4, cn_stop4, cn_weight4, cn_type4)]
        for config in image_prompt_config:
            cn_img, cn_stop, cn_weight, cn_type = config
            if cn_stop is None:
                cn_stop = flags.default_parameters[cn_type.value][0]
            if cn_weight is None:
                cn_weight = flags.default_parameters[cn_type.value][1]
            image_prompts.append(ImagePrompt(
                cn_img=cn_img, cn_stop=cn_stop, cn_weight=cn_weight, cn_type=cn_type))

        style_selection_arr: List[str] = []
        for part in style_selections:
            if len(part) > 0:
                for s in part.split(','):
                    style = s.strip()
                    style_selection_arr.append(style)

        loras: List[Lora] = []
        lora_config = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]
        for config in lora_config:
            lora_model, lora_weight = config
            if lora_model is not None and len(lora_model) > 0:
                loras.append(Lora(model_name=lora_model, weight=lora_weight))

        advanced_params_obj = None
        if advanced_params != None and len(advanced_params) > 0:
            try:
                advanced_params_obj = AdvancedParams.__pydantic_validator__.validate_json(advanced_params)
            except ValidationError as ve:
                errs = ve.errors()
                raise RequestValidationError(errors=[errs])

        return cls(image_prompts=image_prompts, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name, refiner_switch=refiner_switch,
                   loras=loras, advanced_params=advanced_params_obj, require_base64=require_base64, async_process=async_process)


class GeneratedImageResult(BaseModel):
    base64: str | None = Field(
        description="Image encoded in base64, or null if finishReasen is not 'SUCCESS', only return when request require base64")
    url: str | None = Field(description="Image file static serve url, or null if finishReasen is not 'SUCCESS'")
    seed: int = Field(description="The seed associated with this image")
    finish_reason: GenerationFinishReason


class AsyncJobStage(str, Enum):
    waiting = 'WAITING'
    running = 'RUNNING'
    success = 'SUCCESS'
    error = 'ERROR'


class AsyncJobResponse(BaseModel):
    job_id: int
    job_type: TaskType
    job_stage: AsyncJobStage
    job_progess: int
    job_status: str | None
    job_result: List[GeneratedImageResult] | None


class JobQueueInfo(BaseModel):
    running_size: int = Field(description="The current running and waiting job count")
    finished_size: int = Field(description="Finished job cound (after auto clean)")
    last_job_id: int = Field(description="Last submit generation job id")


class AllModelNamesResponse(BaseModel):
    model_filenames: List[str]
    lora_filenames: List[str]

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )