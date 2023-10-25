from fastapi import Form, UploadFile
from fastapi.params import File
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from typing import List
from enum import Enum

from pydantic_core import InitErrorDetails
from fooocusapi.parameters import GenerationFinishReason, defualt_styles
from fooocusapi.task_queue import TaskType
import modules.flags as flags


class Lora(BaseModel):
    model_name: str
    weight: float = Field(default=0.5, min=-2, max=2)

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )


class PerfomanceSelection(str, Enum):
    speed = 'Speed'
    quality = 'Quality'


class AspectRatio(str, Enum):
    a_0_5 = '704×1408'
    a_0_52 = '704×1344'
    a_0_57 = '768×1344'
    a_0_6 = '768×1280'
    a_0_68 = '832×1216'
    a_0_72 = '832×1152'
    a_0_78 = '896×1152'
    a_0_82 = '896×1088'
    a_0_88 = '960×1088'
    a_0_94 = '960×1024'
    a_1_0 = '1024×1024'
    a_1_07 = '1024×960'
    a_1_13 = '1088×960'
    a_1_21 = '1088×896'
    a_1_29 = '1152×896'
    a_1_38 = '1152×832'
    a_1_46 = '1216×832'
    a_1_67 = '1280×768'
    a_1_75 = '1344×768'
    a_1_91 = '1344×704'
    a_2_0 = '1408×704'
    a_2_09 = '1472×704'
    a_2_4 = '1536×640'
    a_2_5 = '1600×640'
    a_2_89 = '1664×576'
    a_3_0 = '1728×576'


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
    cn_stop: float = Field(default=0.4, min=0, max=1)
    cn_weight: float | None = Field(
        default=None, min=0, max=2, description="None for default value")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip)


class Text2ImgRequest(BaseModel):
    prompt: str = ''
    negative_prompt: str = ''
    style_selections: List[str] = defualt_styles
    performance_selection: PerfomanceSelection = PerfomanceSelection.speed
    aspect_ratios_selection: AspectRatio = AspectRatio.a_1_29
    image_number: int = Field(
        default=1, description="Image number", min=1, max=32)
    image_seed: int = Field(default=-1, description="Seed to generate image, -1 for random")
    sharpness: float = Field(default=2.0, min=0.0, max=30.0)
    guidance_scale: float = Field(default=7.0, min=1.0, max=30.0)
    base_model_name: str = 'sd_xl_base_1.0_0.9vae.safetensors'
    refiner_model_name: str = 'sd_xl_refiner_1.0_0.9vae.safetensors'
    loras: List[Lora] = Field(default=[
        Lora(model_name='sd_xl_offset_example-lora_1.0.safetensors', weight=0.5)])
    async_process: bool = Field(default=False, description="Set to true will run async and return job info for retrieve generataion result later")


class ImgUpscaleOrVaryRequest(Text2ImgRequest):
    input_image: UploadFile
    uov_method: UpscaleOrVaryMethod

    @classmethod
    def as_form(cls, input_image: UploadFile = Form(description="Init image for upsacale or outpaint"),
                uov_method: UpscaleOrVaryMethod = Form(),
                prompt: str = Form(''),
                negative_prompt: str = Form(''),
                style_selections: List[str] = Form(defualt_styles, description="Fooocus style selections, seperated by comma"),
                performance_selection: PerfomanceSelection = Form(
                    PerfomanceSelection.speed),
                aspect_ratios_selection: AspectRatio = Form(
                    AspectRatio.a_1_29),
                image_number: int = Form(
                    default=1, description="Image number", ge=1, le=32),
                image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
                sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
                guidance_scale: float = Form(default=7.0, ge=1.0, le=30.0),
                base_model_name: str = Form(
                    'sd_xl_base_1.0_0.9vae.safetensors'),
                refiner_model_name: str = Form(
                    'sd_xl_refiner_1.0_0.9vae.safetensors'),
                l1: str | None = Form(
                    'sd_xl_offset_example-lora_1.0.safetensors'),
                w1: float = Form(default=0.5, ge=-2, le=2),
                l2: str | None = Form(None),
                w2: float = Form(default=0.5, ge=-2, le=2),
                l3: str | None = Form(None),
                w3: float = Form(default=0.5, ge=-2, le=2),
                l4: str | None = Form(None),
                w4: float = Form(default=0.5, ge=-2, le=2),
                l5: str | None = Form(None),
                w5: float = Form(default=0.5, ge=-2, le=2),
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

        return cls(input_image=input_image, uov_method=uov_method, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name,
                   loras=loras, async_process=async_process)


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
                negative_prompt: str = Form(''),
                style_selections: List[str] = Form(defualt_styles, description="Fooocus style selections, seperated by comma"),
                performance_selection: PerfomanceSelection = Form(
                    PerfomanceSelection.speed),
                aspect_ratios_selection: AspectRatio = Form(
                    AspectRatio.a_1_29),
                image_number: int = Form(
                    default=1, description="Image number", ge=1, le=32),
                image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
                sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
                guidance_scale: float = Form(default=7.0, ge=1.0, le=30.0),
                base_model_name: str = Form(
                    'sd_xl_base_1.0_0.9vae.safetensors'),
                refiner_model_name: str = Form(
                    'sd_xl_refiner_1.0_0.9vae.safetensors'),
                l1: str | None = Form(
                    'sd_xl_offset_example-lora_1.0.safetensors'),
                w1: float = Form(default=0.5, ge=-2, le=2),
                l2: str | None = Form(None),
                w2: float = Form(default=0.5, ge=-2, le=2),
                l3: str | None = Form(None),
                w3: float = Form(default=0.5, ge=-2, le=2),
                l4: str | None = Form(None),
                w4: float = Form(default=0.5, ge=-2, le=2),
                l5: str | None = Form(None),
                w5: float = Form(default=0.5, ge=-2, le=2),
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

        return cls(input_image=input_image, input_mask=input_mask, outpaint_selections=outpaint_selections_arr, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name,
                   loras=loras, async_process=async_process)


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
                negative_prompt: str = Form(''),
                style_selections: List[str] = Form(defualt_styles, description="Fooocus style selections, seperated by comma"),
                performance_selection: PerfomanceSelection = Form(
                    PerfomanceSelection.speed),
                aspect_ratios_selection: AspectRatio = Form(
                    AspectRatio.a_1_29),
                image_number: int = Form(
                    default=1, description="Image number", ge=1, le=32),
                image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
                sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
                guidance_scale: float = Form(default=7.0, ge=1.0, le=30.0),
                base_model_name: str = Form(
                    'sd_xl_base_1.0_0.9vae.safetensors'),
                refiner_model_name: str = Form(
                    'sd_xl_refiner_1.0_0.9vae.safetensors'),
                l1: str | None = Form(
                    'sd_xl_offset_example-lora_1.0.safetensors'),
                w1: float = Form(default=0.5, ge=-2, le=2),
                l2: str | None = Form(None),
                w2: float = Form(default=0.5, ge=-2, le=2),
                l3: str | None = Form(None),
                w3: float = Form(default=0.5, ge=-2, le=2),
                l4: str | None = Form(None),
                w4: float = Form(default=0.5, ge=-2, le=2),
                l5: str | None = Form(None),
                w5: float = Form(default=0.5, ge=-2, le=2),
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

        return cls(image_prompts=image_prompts, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name,
                   loras=loras, async_process=async_process)


class GeneratedImageBase64(BaseModel):
    base64: str | None = Field(
        description="Image encoded in base64, or null if finishReasen is not 'SUCCESS'")
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
    job_result: List[GeneratedImageBase64] | None


class JobQueueInfo(BaseModel):
    running_size: int = Field(description="The current running and waiting job count")
    finished_size: int = Field(description="Finished job cound (after auto clean)")
    last_job_id: int = Field(description="Last submit generation job id")


class AllModelNamesResponse(BaseModel):
    model_filenames: List[str]
    lora_filenames: List[str]