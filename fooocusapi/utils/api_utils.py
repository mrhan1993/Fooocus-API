"""API utils"""
from fastapi.security import APIKeyHeader
from fastapi import HTTPException, Security

from modules import flags
from modules import config
from modules.sdxl_styles import legal_style_names

from fooocusapi.args import args
from fooocusapi.utils.img_utils import read_input_image

from fooocusapi.models.common.task import TaskType
from fooocusapi.models.common.base import CommonRequest as Text2ImgRequest
from fooocusapi.models.v1.requests import (
    ImgInpaintOrOutpaintRequest,
    ImgPromptRequest,
    ImgUpscaleOrVaryRequest,
)

from fooocusapi.models.v2.request import (
    ImgInpaintOrOutpaintRequestJson,
    ImgPromptRequestJson,
    Text2ImgRequestWithPrompt,
    ImgUpscaleOrVaryRequestJson)

from fooocusapi.parameters import (
    ImageGenerationParams,
    default_inpaint_engine_version,
    default_sampler,
    default_scheduler,
    default_base_model_name,
    default_refiner_model_name
)


img_generate_responses = {
    "200": {
        "description": "PNG bytes if request's 'Accept' header is 'image/png', otherwise JSON",
        "content": {
            "application/json": {
                "example": [
                    {
                        "base64": "...very long string...",
                        "seed": "1050625087",
                        "finish_reason": "SUCCESS",
                    }
                ]
            },
            "application/json async": {
                "example": {"job_id": 1, "job_type": "Text to Image"}
            },
            "image/png": {"example": "PNG bytes, what did you expect?"},
        },
    }
}

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

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

    # Check base_model_name
    if req.base_model_name in config.model_filenames:
        pass
    else:
        print(f"[Warning] Wrong base_model_name input: {req.base_model_name}, using default")
        req.base_model_name = default_base_model_name

    # Check refiner_model_name
    if req.refiner_model_name in config.default_refiner_model_name:
        pass
    else:
        print(f"[Warning] Wrong refiner_model_name input: {req.refiner_model_name}, using default")
        req.refiner_model_name = default_refiner_model_name

    # Check loras
    loras = []
    for lora in req.loras:
        if lora.model_name in config.lora_filenames:
            loras.append((lora.model_name, lora.weight))
        else:
            loras.append(('None', lora.weight))
    req.loras = loras

    # convert and check style_selections
    req.style_selections = [
        s for s in req.style_selections if s in legal_style_names]

    req.image_seed = None if req.image_seed == -1 else req.image_seed

    # parse uov params
    uov_input_image = None
    if not isinstance(req, Text2ImgRequestWithPrompt):
        if isinstance(req, (ImgUpscaleOrVaryRequest, ImgUpscaleOrVaryRequestJson)):
            uov_input_image = read_input_image(req.input_image)
            uov_method = req.uov_method.value
            upscale_value = req.upscale_value
        else:
            uov_method = flags.disabled
            upscale_value = None
    else:
        uov_method = flags.disabled
        upscale_value = None

    # parse outpaint params
    if not isinstance(req, (ImgInpaintOrOutpaintRequest, ImgInpaintOrOutpaintRequestJson)):
        outpaint_selections = []
        outpaint_distance_left = None
        outpaint_distance_right = None
        outpaint_distance_top = None
        outpaint_distance_bottom = None
    else:
        outpaint_selections = [s.value for s in req.outpaint_selections]
        outpaint_distance_left = req.outpaint_distance_left
        outpaint_distance_right = req.outpaint_distance_right
        outpaint_distance_top = req.outpaint_distance_top
        outpaint_distance_bottom = req.outpaint_distance_bottom

    # parse inpaint params
    inpaint_input_image = None
    inpaint_additional_prompt = None
    if isinstance(req, (ImgInpaintOrOutpaintRequest, ImgInpaintOrOutpaintRequestJson)):
        inpaint_additional_prompt = req.inpaint_additional_prompt
        input_image = read_input_image(req.input_image)
        input_mask = read_input_image(req.input_mask)
        inpaint_input_image = {
            'image': input_image,
            'mask': input_mask
        }

    # parse image prompts params
    image_prompts = []
    if isinstance(req, (ImgInpaintOrOutpaintRequestJson, ImgPromptRequest,
                        ImgPromptRequestJson, ImgUpscaleOrVaryRequestJson,
                        Text2ImgRequestWithPrompt)):
        # Auto set mixing_image_prompt_and_inpaint to True
        if len(req.image_prompts) > 0 and req.uov_input_image is not None:
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

    # parse and check advanced_params
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
            adp.disable_preview, adp.adm_scaler_positive, adp.adm_scaler_negative,
            adp.adm_scaler_end, adp.adaptive_cfg, adp.sampler_name,
            adp.scheduler_name, False, adp.overwrite_step,
            adp.overwrite_switch, adp.overwrite_width, adp.overwrite_height,
            adp.overwrite_vary_strength, adp.overwrite_upscale_strength,
            adp.mixing_image_prompt_and_vary_upscale, adp.mixing_image_prompt_and_inpaint,
            adp.debugging_cn_preprocessor, adp.skipping_cn_preprocessor, adp.controlnet_softness,
            adp.canny_low_threshold, adp.canny_high_threshold,
            adp.refiner_swap_method,
            adp.freeu_enabled, adp.freeu_b1, adp.freeu_b2, adp.freeu_s1, adp.freeu_s2,
            adp.debugging_inpaint_preprocessor, adp.inpaint_disable_initial_latent,
            adp.inpaint_engine, adp.inpaint_strength, adp.inpaint_respective_field,
            False, adp.invert_mask_checkbox, adp.inpaint_erode_or_dilate
        ]

    return ImageGenerationParams(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        style_selections=req.style_selections,
        performance_selection=req.performance_selection.value,
        aspect_ratios_selection=req.aspect_ratios_selection,
        image_number=req.image_number,
        image_seed=req.image_seed,
        sharpness=req.sharpness,
        guidance_scale=req.guidance_scale,
        base_model_name=req.base_model_name,
        refiner_model_name=req.refiner_model_name,
        refiner_switch=req.refiner_switch,
        loras=req.loras,

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
        require_base64=req.require_base64,
    )


def get_task_type(req: Text2ImgRequest) -> TaskType:
    """
    Get task type from request
    Args:
        req: Text2ImgRequest, or other object inherit CommonRequest
    returns: TaskType
    """
    if isinstance(req, (ImgUpscaleOrVaryRequest, ImgUpscaleOrVaryRequestJson)):
        return TaskType.img_uov
    if isinstance(req, (ImgPromptRequest, ImgPromptRequestJson)):
        return TaskType.img_prompt
    if isinstance(req, (ImgInpaintOrOutpaintRequest, ImgInpaintOrOutpaintRequestJson)):
        return TaskType.img_inpaint_outpaint
    return TaskType.text_2_img
