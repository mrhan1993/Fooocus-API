from typing import List

from fastapi import Response
from fooocusapi.file_utils import get_file_serve_url, output_file_to_base64img, output_file_to_bytesimg
from fooocusapi.img_utils import read_input_image
from fooocusapi.models import AsyncJobResponse, AsyncJobStage, GeneratedImageResult, GenerationFinishReason, ImgInpaintOrOutpaintRequest, ImgPromptRequest, ImgUpscaleOrVaryRequest, Text2ImgRequest
from fooocusapi.models_v2 import *
from fooocusapi.parameters import ImageGenerationParams, ImageGenerationResult, default_inpaint_engine_version, default_sampler, default_scheduler, default_base_model_name, default_refiner_model_name
from fooocusapi.task_queue import QueueTask

from modules import flags
from modules import config
from modules.sdxl_styles import legal_style_names


def req_to_params(req: Text2ImgRequest) -> ImageGenerationParams:
    if req.base_model_name is not None:
        if req.base_model_name not in config.model_filenames:
            print(f"[Warning] Wrong base_model_name input: {req.base_model_name}, using default")
            req.base_model_name = default_base_model_name

    if req.refiner_model_name is not None and req.refiner_model_name != 'None':
        if req.refiner_model_name not in config.model_filenames:
            print(f"[Warning] Wrong refiner_model_name input: {req.refiner_model_name}, using default")
            req.refiner_model_name = default_refiner_model_name

    for l in req.loras:
        if l.model_name != 'None' and l.model_name not in config.lora_filenames:
            print(f"[Warning] Wrong lora model_name input: {l.model_name}, using 'None'")
            l.model_name = 'None'

    prompt = req.prompt
    negative_prompt = req.negative_prompt
    style_selections = [
        s for s in req.style_selections if s in legal_style_names]
    performance_selection = req.performance_selection.value
    aspect_ratios_selection = req.aspect_ratios_selection
    image_number = req.image_number
    image_seed = None if req.image_seed == -1 else req.image_seed
    sharpness = req.sharpness
    guidance_scale = req.guidance_scale
    base_model_name = req.base_model_name
    refiner_model_name = req.refiner_model_name
    refiner_switch = req.refiner_switch
    loras = [(lora.model_name, lora.weight) for lora in req.loras]
    uov_input_image = None
    if not isinstance(req, Text2ImgRequestWithPrompt):
        if isinstance(req, ImgUpscaleOrVaryRequest) or isinstance(req, ImgUpscaleOrVaryRequestJson):
            uov_input_image = read_input_image(req.input_image)
    uov_method = flags.disabled if not (isinstance(
        req, ImgUpscaleOrVaryRequest) or isinstance(req, ImgUpscaleOrVaryRequestJson)) else req.uov_method.value
    upscale_value = None if not (isinstance(
                req, ImgUpscaleOrVaryRequest) or isinstance(req, ImgUpscaleOrVaryRequestJson)) else req.upscale_value
    outpaint_selections = [] if not (isinstance(
        req, ImgInpaintOrOutpaintRequest) or isinstance(req, ImgInpaintOrOutpaintRequestJson)) else [
        s.value for s in req.outpaint_selections]
    outpaint_distance_left = None if not (isinstance(
        req, ImgInpaintOrOutpaintRequest) or isinstance(req, ImgInpaintOrOutpaintRequestJson)) else req.outpaint_distance_left
    outpaint_distance_right = None if not (isinstance(
        req, ImgInpaintOrOutpaintRequest) or isinstance(req, ImgInpaintOrOutpaintRequestJson)) else req.outpaint_distance_right
    outpaint_distance_top = None if not (isinstance(
        req, ImgInpaintOrOutpaintRequest) or isinstance(req, ImgInpaintOrOutpaintRequestJson)) else req.outpaint_distance_top
    outpaint_distance_bottom = None if not (isinstance(
        req, ImgInpaintOrOutpaintRequest) or isinstance(req, ImgInpaintOrOutpaintRequestJson)) else req.outpaint_distance_bottom

    if refiner_model_name == '':
        refiner_model_name = 'None'

    inpaint_input_image = None
    inpaint_additional_prompt = None
    if (isinstance(req, ImgInpaintOrOutpaintRequest) or isinstance(req, ImgInpaintOrOutpaintRequestJson)) and req.input_image is not None:
        inpaint_additional_prompt = req.inpaint_additional_prompt
        input_image = read_input_image(req.input_image)
        input_mask = None
        if req.input_mask is not None:
            input_mask = read_input_image(req.input_mask)
        inpaint_input_image = {
            'image': input_image,
            'mask': input_mask
        }

    image_prompts = []
    if isinstance(req, ImgPromptRequest) or isinstance(req, ImgPromptRequestJson) or isinstance(req, Text2ImgRequestWithPrompt):
        # Auto set mixing_image_prompt_and_inpaint to True
        if len(req.image_prompts) > 0 and not isinstance(req, Text2ImgRequestWithPrompt) and req.input_image is not None and req.advanced_params is not None:
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
        
        advanced_params = [
            adp.disable_preview, adp.adm_scaler_positive, adp.adm_scaler_negative, adp.adm_scaler_end, adp.adaptive_cfg, adp.sampler_name, \
            adp.scheduler_name, False, adp.overwrite_step, adp.overwrite_switch, adp.overwrite_width, adp.overwrite_height, \
            adp.overwrite_vary_strength, adp.overwrite_upscale_strength, \
            adp.mixing_image_prompt_and_vary_upscale, adp.mixing_image_prompt_and_inpaint, \
            adp.debugging_cn_preprocessor, adp.skipping_cn_preprocessor, adp.controlnet_softness, adp.canny_low_threshold, adp.canny_high_threshold, \
            adp.refiner_swap_method, \
            adp.freeu_enabled, adp.freeu_b1, adp.freeu_b2, adp.freeu_s1, adp.freeu_s2, \
            adp.debugging_inpaint_preprocessor, adp.inpaint_disable_initial_latent, adp.inpaint_engine, adp.inpaint_strength, adp.inpaint_respective_field, \
            False, adp.invert_mask_checkbox, adp.inpaint_erode_or_dilate
        ]

    return ImageGenerationParams(prompt=prompt,
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
                                 image_prompts=image_prompts,
                                 advanced_params=advanced_params,
                                 )


def generation_output(results: QueueTask | List[ImageGenerationResult], streaming_output: bool, require_base64: bool, require_step_preivew: bool=False) -> Response | List[GeneratedImageResult] | AsyncJobResponse:
    if isinstance(results, QueueTask):
        task = results
        job_stage = AsyncJobStage.running
        job_result = None
        if task.start_millis == 0:
            job_stage = AsyncJobStage.waiting
        if task.is_finished:
            if task.finish_with_error:
                job_stage = AsyncJobStage.error
            else:
                if task.task_result != None:
                    job_stage = AsyncJobStage.success
                    task_result_require_base64 = False
                    if 'require_base64' in task.req_param and task.req_param['require_base64']:
                        task_result_require_base64 = True

                    job_result = generation_output(task.task_result, False, task_result_require_base64)
        job_step_preview = None if not require_step_preivew else task.task_step_preview
        return AsyncJobResponse(job_id=task.job_id,
                                job_type=task.type,
                                job_stage=job_stage,
                                job_progress=task.finish_progress,
                                job_status=task.task_status,
                                job_step_preview=job_step_preview,
                                job_result=job_result)

    if streaming_output:
        if len(results) == 0:
            return Response(status_code=500)
        result = results[0]
        if result.finish_reason == GenerationFinishReason.queue_is_full:
            return Response(status_code=409, content=result.finish_reason.value)
        elif result.finish_reason == GenerationFinishReason.user_cancel:
            return Response(status_code=400, content=result.finish_reason.value)
        elif result.finish_reason == GenerationFinishReason.error:
            return Response(status_code=500, content=result.finish_reason.value)
        
        bytes = output_file_to_bytesimg(results[0].im)
        return Response(bytes, media_type='image/png')
    else:
        results = [GeneratedImageResult(
            base64=output_file_to_base64img(
                item.im) if require_base64 else None,
            url=get_file_serve_url(item.im),
            seed=item.seed,
            finish_reason=item.finish_reason) for item in results]
        return results


class QueueReachLimitException(Exception):
    pass
