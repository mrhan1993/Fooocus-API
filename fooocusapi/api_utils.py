import base64
import io
from io import BytesIO
from typing import List

import numpy as np
from fastapi import Response, UploadFile
from PIL import Image
from fooocusapi.models import GeneratedImageBase64, GenerationFinishReason, ImgInpaintOrOutpaintRequest, ImgPromptRequest, ImgUpscaleOrVaryRequest, Text2ImgRequest
from fooocusapi.parameters import ImageGenerationParams, ImageGenerationResult
import modules.flags as flags


def narray_to_base64img(narray: np.ndarray) -> str:
    if narray is None:
        return None

    img = Image.fromarray(narray)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def narray_to_bytesimg(narray) -> bytes:
    if narray is None:
        return None

    img = Image.fromarray(narray)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    return byte_data


def read_input_image(input_image: UploadFile) -> np.ndarray:
    input_image_bytes = input_image.file.read()
    pil_image = Image.open(io.BytesIO(input_image_bytes))
    image = np.array(pil_image)
    return image


def req_to_params(req: Text2ImgRequest) -> ImageGenerationParams:
    prompt = req.prompt
    negative_prompt = req.negative_prompt
    style_selections = [s.value for s in req.style_selections]
    performance_selection = req.performance_selection.value
    aspect_ratios_selection = req.aspect_ratios_selection.value
    image_number = req.image_number
    image_seed = None if req.image_seed == -1 else req.image_seed
    sharpness = req.sharpness
    guidance_scale = req.guidance_scale
    base_model_name = req.base_model_name
    refiner_model_name = req.refiner_model_name
    loras = [(lora.model_name, lora.weight) for lora in req.loras]
    uov_input_image = None if not isinstance(
        req, ImgUpscaleOrVaryRequest) else read_input_image(req.input_image)
    uov_method = flags.disabled if not isinstance(
        req, ImgUpscaleOrVaryRequest) else req.uov_method.value
    outpaint_selections = [] if not isinstance(req, ImgInpaintOrOutpaintRequest) else [
        s.value for s in req.outpaint_selections]

    inpaint_input_image = None
    if isinstance(req, ImgInpaintOrOutpaintRequest):
        input_image = read_input_image(req.input_image)
        if req.input_mask is not None:
            input_mask = read_input_image(req.input_mask)
        else:
            input_mask = np.zeros(input_image.shape)
        inpaint_input_image = {
            'image': input_image,
            'mask': input_mask
        }

    image_prompts = []
    if isinstance(req, ImgPromptRequest):
        for img_prompt in req.image_prompts:
            if img_prompt.cn_img is not None:
                cn_img = read_input_image(img_prompt.cn_img)
                image_prompts.append(
                    (cn_img, img_prompt.cn_stop, img_prompt.cn_weight, img_prompt.cn_type.value))

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
                                 loras=loras,
                                 uov_input_image=uov_input_image,
                                 uov_method=uov_method,
                                 outpaint_selections=outpaint_selections,
                                 inpaint_input_image=inpaint_input_image,
                                 image_prompts=image_prompts
                                 )


def generation_output(results: List[ImageGenerationResult], streaming_output: bool) -> Response | List[GeneratedImageBase64]:
    if streaming_output:
        if len(results) == 0 or results[0].finish_reason != GenerationFinishReason.success:
            return Response(status_code=500)
        bytes = narray_to_bytesimg(results[0].im)
        return Response(bytes, media_type='image/png')
    else:
        results = [GeneratedImageBase64(base64=narray_to_base64img(
            item.im), seed=item.seed, finish_reason=item.finish_reason) for item in results]
        return results


class QueueReachLimitException(Exception):
    pass
