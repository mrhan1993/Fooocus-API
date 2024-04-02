"""Generate API V2 routes

"""
from typing import List
from fastapi import APIRouter, Depends, Header, Query

from fooocusapi.api_utils import api_key_auth
from fooocusapi.models.requests_v1 import ImagePrompt
from fooocusapi.models.requests_v2 import (
    ImgInpaintOrOutpaintRequestJson,
    ImgPromptRequestJson,
    Text2ImgRequestWithPrompt,
    ImgUpscaleOrVaryRequestJson
)
from fooocusapi.models.common.response import (
    AsyncJobResponse,
    GeneratedImageResult
)
from fooocusapi.utils.call_worker import call_worker
from fooocusapi.utils.img_utils import base64_to_stream
from fooocusapi.parameters import img_generate_responses


secure_router = APIRouter(
    dependencies=[Depends(api_key_auth)]
)


@secure_router.post(
        path="/v2/generation/text-to-image-with-ip",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def text_to_img_with_ip(
    req: Text2ImgRequestWithPrompt,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    """\nText to image with prompt\n
    Text to image with prompt
    Arguments:
        req {Text2ImgRequestWithPrompt} -- Text to image generation request
        accept {str} -- Accept header
        accept_query {str} -- Parameter to overvide 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)

    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)

    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post(
        path="/v2/generation/image-upscale-vary",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_upscale_or_vary(
    req: ImgUpscaleOrVaryRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    """\nImage upscale or vary\n
    Image upscale or vary
    Arguments:
        req {ImgUpscaleOrVaryRequestJson} -- Image upscale or vary request
        accept {str} -- Accept header
        accept_query {str} -- Parameter to overvide 'Accept' header, 'image/png' for output bytes
    Returns:
            Response -- img_generate_responses    
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    req.input_image = base64_to_stream(req.input_image)

    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)
    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post(
        path="/v2/generation/image-inpaint-outpaint",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_inpaint_or_outpaint(
    req: ImgInpaintOrOutpaintRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    """\nInpaint or outpaint\n
    Inpaint or outpaint
    Arguments:
        req {ImgInpaintOrOutpaintRequestJson} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to overvide 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)
    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)
    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post(
        path="/v2/generation/image-prompt",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_prompt(
    req: ImgPromptRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    """\nImage prompt\n
    Image prompt generation
    Arguments:
        req {ImgPromptRequest} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to overvide 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if req.input_image is not None:
        req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)

    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)

    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)

    req.image_prompts = image_prompts_files

    return call_worker(req, accept)
