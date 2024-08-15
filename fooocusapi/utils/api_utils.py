"""some utils for api"""
import random
from typing import List

import numpy
from fastapi import Response
from fastapi.security import APIKeyHeader
from fastapi import HTTPException, Security

from fooocusapi.models.common.base import EnhanceCtrlNets, ImagePrompt
from modules import constants, flags
from modules import config
from modules.sdxl_styles import legal_style_names

from fooocusapi.args import args
from fooocusapi.utils.img_utils import read_input_image
from fooocusapi.utils.file_utils import (
    get_file_serve_url,
    output_file_to_base64img,
    output_file_to_bytesimg
)
from fooocusapi.utils.logger import logger
from fooocusapi.models.common.requests import (
    CommonRequest as Text2ImgRequest
)
from fooocusapi.models.common.response import (
    AsyncJobResponse,
    AsyncJobStage,
    GeneratedImageResult
)
from fooocusapi.models.requests_v1 import (
    ImageEnhanceRequest, ImgInpaintOrOutpaintRequest,
    ImgPromptRequest,
    ImgUpscaleOrVaryRequest
)
from fooocusapi.models.requests_v2 import (
    ImageEnhanceRequestJson, Text2ImgRequestWithPrompt,
    ImgInpaintOrOutpaintRequestJson,
    ImgUpscaleOrVaryRequestJson,
    ImgPromptRequestJson
)
from fooocusapi.models.common.task import (
    ImageGenerationResult,
    GenerationFinishReason
)
from fooocusapi.configs.default import (
    default_inpaint_engine_version,
    default_sampler,
    default_scheduler,
    default_base_model_name,
    default_refiner_model_name
)

from fooocusapi.parameters import ImageGenerationParams
from fooocusapi.task_queue import QueueTask
from modules.util import HWC3

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


def refresh_seed(seed_string: int | str | None) -> int:
    """
    Refresh and check seed number.
    :params seed_string: seed, str or int. None means random
    :return: seed number
    """
    RANDOM_SEED = random.randint(constants.MIN_SEED, constants.MAX_SEED)
    try:
        seed_value = int(seed_string)
    except ValueError:
        return RANDOM_SEED

    if seed_value < constants.MIN_SEED or seed_value > constants.MAX_SEED or seed_string == -1:
        return RANDOM_SEED

    return seed_value


def check_models_exist(file_name: str, model_type: str) -> str:
    """
    Check if all models exist
    """
    if file_name in (None, 'None'):
        return 'None'

    config.update_files()
    if file_name not in (config.model_filenames + config.lora_filenames):
        logger.std_warn(f"[Warning] Wrong {model_type} model input: {file_name}, using default")
        if model_type == 'base':
            return default_base_model_name
        if model_type == 'refiner':
            return default_refiner_model_name
        return 'None'
    return file_name


def api_key_auth(apikey: str = Security(api_key_header)):
    """
    Check if the API key is valid, API key is not required if no API key is set
    Args:
        apikey: API key
    returns:
        None if API key is not set, otherwise raise HTTPException
    """
    if args.apikey is None:
        return  # Skip API key check if no API key is set
    if apikey != args.apikey:
        raise HTTPException(status_code=403, detail="Forbidden")


def req_to_params(req: Text2ImgRequest) -> ImageGenerationParams:
    """
    Convert Request to ImageGenerationParams
    Args:
        req: Request, Text2ImgRequest and classes inherited from Text2ImgRequest
    returns:
        ImageGenerationParams
    """
    prompt = req.prompt
    negative_prompt = req.negative_prompt
    style_selections = [
        s for s in req.style_selections if s in legal_style_names]
    performance_selection = req.performance_selection.value
    aspect_ratios_selection = req.aspect_ratios_selection
    image_number = req.image_number
    image_seed = refresh_seed(req.image_seed)
    sharpness = req.sharpness
    guidance_scale = req.guidance_scale
    base_model_name = check_models_exist(req.base_model_name, 'base')
    refiner_model_name = check_models_exist(req.refiner_model_name, 'refiner')
    refiner_switch = req.refiner_switch
    loras = [(lora.enabled, check_models_exist(lora.model_name, 'lora'), lora.weight) for lora in req.loras]
    uov_input_image = None
    if not isinstance(req, Text2ImgRequestWithPrompt):
        if isinstance(req, (ImgUpscaleOrVaryRequest, ImgUpscaleOrVaryRequestJson)):
            uov_input_image = read_input_image(req.input_image)
    uov_method = flags.disabled if not isinstance(req, (ImgUpscaleOrVaryRequest, ImgUpscaleOrVaryRequestJson)) else req.uov_method.value
    upscale_value = None if not isinstance(req, (ImgUpscaleOrVaryRequest, ImgUpscaleOrVaryRequestJson)) else req.upscale_value
    outpaint_selections = [] if not isinstance(req, (ImgInpaintOrOutpaintRequest, ImgInpaintOrOutpaintRequestJson)) else [
        s.value for s in req.outpaint_selections]
    outpaint_distance_left = 0 if not isinstance(req, (ImgInpaintOrOutpaintRequest, ImgInpaintOrOutpaintRequestJson)) else req.outpaint_distance_left
    outpaint_distance_right = 0 if not isinstance(req, (ImgInpaintOrOutpaintRequest, ImgInpaintOrOutpaintRequestJson)) else req.outpaint_distance_right
    outpaint_distance_top = 0 if not isinstance(req, (ImgInpaintOrOutpaintRequest, ImgInpaintOrOutpaintRequestJson)) else req.outpaint_distance_top
    outpaint_distance_bottom = 0 if not isinstance(req, (ImgInpaintOrOutpaintRequest, ImgInpaintOrOutpaintRequestJson)) else req.outpaint_distance_bottom

    if refiner_model_name == '':
        refiner_model_name = 'None'

    inpaint_input_image = dict(image=None, mask=None)
    inpaint_additional_prompt = None
    if isinstance(req, (ImgInpaintOrOutpaintRequest, ImgInpaintOrOutpaintRequestJson)) and req.input_image is not None:
        inpaint_additional_prompt = req.inpaint_additional_prompt
        input_image = read_input_image(req.input_image)

        inpaint_image_size = input_image.shape[:2]
        input_mask = HWC3(numpy.zeros(inpaint_image_size, dtype=numpy.uint8))
        if req.input_mask is not None:
            input_mask = HWC3(read_input_image(req.input_mask))

        inpaint_input_image = {
            'image': input_image,
            'mask': input_mask
        }

    image_prompts = []
    if isinstance(req, (ImgInpaintOrOutpaintRequestJson, ImgPromptRequest, ImgPromptRequestJson, ImgUpscaleOrVaryRequestJson, Text2ImgRequestWithPrompt)):
        # Auto set mixing_image_prompt_and_inpaint to True
        if len(req.image_prompts) > 0 and uov_input_image is not None:
            print("[INFO] Mixing image prompt and vary upscale is set to True")
            req.advanced_params.mixing_image_prompt_and_vary_upscale = True
        elif len(req.image_prompts) > 0 and not isinstance(req, Text2ImgRequestWithPrompt) and req.input_image is not None:
            print("[INFO] Mixing image prompt and inpaint is set to True")
            req.advanced_params.mixing_image_prompt_and_inpaint = True

        for img_prompt in req.image_prompts:
            if img_prompt.cn_img is not None:
                cn_img = read_input_image(img_prompt.cn_img)
                if img_prompt.cn_stop is None or img_prompt.cn_stop == 0:
                    img_prompt.cn_stop = flags.default_parameters[img_prompt.cn_type.value][0]
                if img_prompt.cn_weight is None or img_prompt.cn_weight == 0:
                    img_prompt.cn_weight = flags.default_parameters[img_prompt.cn_type.value][1]
                image_prompts.append(
                    (cn_img, img_prompt.cn_stop, img_prompt.cn_weight, img_prompt.cn_type.value))

    if len(image_prompts) < config.default_controlnet_image_count:
        dp = (None, 0.5, 0.6, 'ImagePrompt')
        image_prompts += [dp] * (config.default_controlnet_image_count - len(image_prompts))

    if isinstance(req, (ImageEnhanceRequest, ImageEnhanceRequestJson)):
        enhance_checkbox = True
        enhance_input_image = read_input_image(req.enhance_input_image)
        enhance_uov_method = req.enhance_uov_method
        enhance_uov_processing_order = req.enhance_uov_processing_order
        enhance_uov_prompt_type = req.enhance_uov_prompt_type
        save_final_enhanced_image_only = True
    else:
        enhance_checkbox = False
        enhance_input_image = None
        enhance_uov_method = flags.disabled
        enhance_uov_processing_order = "Before First Enhancement"
        enhance_uov_prompt_type = "Original Prompts"
        save_final_enhanced_image_only = False

    if not isinstance(req, (ImageEnhanceRequest, ImageEnhanceRequestJson)):
        enhance_ctrlnets = [EnhanceCtrlNets()] * config.default_enhance_tabs
    else:
        enhance_ctrlnets = req.enhance_ctrlnets

    advanced_params = None
    if req.advanced_params is not None:
        adp = req.advanced_params

        if adp.refiner_swap_method not in ['joint', 'separate', 'vae']:
            print(f"[Warning] Wrong refiner_swap_method input: {adp.refiner_swap_method}, using default")
            adp.refiner_swap_method = 'joint'

        if adp.sampler_name not in flags.sampler_list:
            print(f"[Warning] Wrong sampler_name input: {adp.sampler_name}, using default")
            adp.sampler_name = default_sampler

        if adp.scheduler_name not in flags.scheduler_list:
            print(f"[Warning] Wrong scheduler_name input: {adp.scheduler_name}, using default")
            adp.scheduler_name = default_scheduler

        if adp.inpaint_engine not in flags.inpaint_engine_versions:
            print(f"[Warning] Wrong inpaint_engine input: {adp.inpaint_engine}, using default")
            adp.inpaint_engine = default_inpaint_engine_version

        advanced_params = adp

    return ImageGenerationParams(
        prompt=prompt,
        negative_prompt=negative_prompt,
        style_selections=style_selections,
        performance_selection=performance_selection,
        aspect_ratios_selection=aspect_ratios_selection,
        image_number=image_number,
        image_seed=image_seed,
        sharpness=sharpness,
        guidance_scale=guidance_scale,
        base_model_name=base_model_name,
        refiner_model_name=refiner_model_name,
        refiner_switch=refiner_switch,
        loras=loras,
        uov_input_image=uov_input_image,
        uov_method=uov_method,
        upscale_value=upscale_value,
        outpaint_selections=outpaint_selections,
        outpaint_distance_left=outpaint_distance_left,
        outpaint_distance_right=outpaint_distance_right,
        outpaint_distance_top=outpaint_distance_top,
        outpaint_distance_bottom=outpaint_distance_bottom,
        inpaint_input_image=inpaint_input_image,
        inpaint_additional_prompt=inpaint_additional_prompt,
        enhance_input_image=enhance_input_image,
        enhance_checkbox=enhance_checkbox,
        enhance_uov_method=enhance_uov_method,
        enhance_uov_processing_order=enhance_uov_processing_order,
        enhance_uov_prompt_type=enhance_uov_prompt_type,
        save_final_enhanced_image_only=save_final_enhanced_image_only,
        enhance_ctrlnets=enhance_ctrlnets,
        read_wildcards_in_order=req.read_wildcards_in_order,
        image_prompts=image_prompts,
        advanced_params=advanced_params,
        save_meta=req.save_meta,
        meta_scheme=req.meta_scheme,
        save_name=req.save_name,
        save_extension=req.save_extension,
        require_base64=req.require_base64,
    )


def generate_async_output(
        task: QueueTask,
        require_step_preview: bool = False) -> AsyncJobResponse:
    """
    Generate output for async job
    Arguments:
        task: QueueTask
        require_step_preview: bool
    Returns:
        AsyncJobResponse
    """
    job_stage = AsyncJobStage.running
    job_result = None

    if task.start_mills == 0:
        job_stage = AsyncJobStage.waiting

    if task.is_finished:
        if task.finish_with_error:
            job_stage = AsyncJobStage.error
        elif task.task_result is not None:
            job_stage = AsyncJobStage.success
            job_result = generate_image_result_output(task.task_result, task.req_param.require_base64)

    result = AsyncJobResponse(
        job_id=task.job_id,
        job_type=task.task_type,
        job_stage=job_stage,
        job_progress=task.finish_progress,
        job_status=task.task_status,
        job_step_preview=task.task_step_preview if require_step_preview else None,
        job_result=job_result)
    return result


def generate_streaming_output(results: List[ImageGenerationResult]) -> Response:
    """
    Generate streaming output for image generation results.
    Args:
        results (List[ImageGenerationResult]): List of image generation results.
    Returns:
        Response: Streaming response object, bytes image.
    """
    if len(results) == 0:
        return Response(status_code=500)
    result = results[0]
    if result.finish_reason == GenerationFinishReason.queue_is_full:
        return Response(status_code=409, content=result.finish_reason.value)
    if result.finish_reason == GenerationFinishReason.user_cancel:
        return Response(status_code=400, content=result.finish_reason.value)
    if result.finish_reason == GenerationFinishReason.error:
        return Response(status_code=500, content=result.finish_reason.value)

    img_bytes = output_file_to_bytesimg(results[0].im)
    return Response(img_bytes, media_type='image/png')


def generate_image_result_output(
        results: List[ImageGenerationResult],
        require_base64: bool) -> List[GeneratedImageResult]:
    """
    Generate image result output
    Arguments:
        results: List[ImageGenerationResult]
        require_base64: bool
    Returns:
        List[GeneratedImageResult]
    """
    results = [
        GeneratedImageResult(
            base64=output_file_to_base64img(item.im) if require_base64 else None,
            url=get_file_serve_url(item.im),
            seed=str(item.seed),
            finish_reason=item.finish_reason
            ) for item in results
        ]
    return results
