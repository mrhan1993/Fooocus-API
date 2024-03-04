"""Generate API"""
from typing import List
from fastapi import APIRouter, Depends, Header, Query
from fooocusapi.models.models import (
    AsyncJobResponse,
    GeneratedImageResult,
    ImagePrompt
)
from fooocusapi.models.v2.request import (
    ImgInpaintOrOutpaintRequestJson,
    ImgPromptRequestJson,
    ImgUpscaleOrVaryRequestJson,
    Text2ImgRequestWithPrompt
)
from fooocusapi.utils.api_utils import (
    img_generate_responses,
    api_key_auth,
    call_worker
)
from fooocusapi.utils.img_utils import base64_to_stream


secure_router = APIRouter(dependencies=[Depends(api_key_auth)])

@secure_router.post(
    "/v2/generation/text-to-image-with-ip",
    response_model=List[GeneratedImageResult] | AsyncJobResponse,
    responses=img_generate_responses,
)
def text_to_img_with_ip(
    req: Text2ImgRequestWithPrompt,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None,
        alias="accept",
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes",
    ),
):
    """Generate image from text prompt"""
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for img_prompt in req.image_prompts:
        img_prompt.cn_img = base64_to_stream(img_prompt.cn_img)
        image = ImagePrompt(
            cn_img=img_prompt.cn_img,
            cn_stop=img_prompt.cn_stop,
            cn_weight=img_prompt.cn_weight,
            cn_type=img_prompt.cn_type,
        )
        image_prompts_files.append(image)

    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)

    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post(
    "/v2/generation/image-upscale-vary",
    response_model=List[GeneratedImageResult] | AsyncJobResponse,
    responses=img_generate_responses,
)
def img_upscale_or_vary_v2(
    req: ImgUpscaleOrVaryRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None,
        alias="accept",
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes",
    ),
):
    """Generate image from text prompt"""
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    req.input_image = base64_to_stream(req.input_image)

    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for img_prompt in req.image_prompts:
        img_prompt.cn_img = base64_to_stream(img_prompt.cn_img)
        image = ImagePrompt(
            cn_img=img_prompt.cn_img,
            cn_stop=img_prompt.cn_stop,
            cn_weight=img_prompt.cn_weight,
            cn_type=img_prompt.cn_type,
        )
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)
    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post(
    "/v2/generation/image-inpaint-outpaint",
    response_model=List[GeneratedImageResult] | AsyncJobResponse,
    responses=img_generate_responses,
)
def img_inpaint_or_outpaint_v2(
    req: ImgInpaintOrOutpaintRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None,
        alias="accept",
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes",
    ),
):
    """Inpaint or outpaint image"""
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)
    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for img_prompt in req.image_prompts:
        img_prompt.cn_img = base64_to_stream(img_prompt.cn_img)
        image = ImagePrompt(
            cn_img=img_prompt.cn_img,
            cn_stop=img_prompt.cn_stop,
            cn_weight=img_prompt.cn_weight,
            cn_type=img_prompt.cn_type,
        )
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)
    req.image_prompts = image_prompts_files

    return call_worker(req, accept)


@secure_router.post(
    "/v2/generation/image-prompt",
    response_model=List[GeneratedImageResult] | AsyncJobResponse,
    responses=img_generate_responses,
)
def img_prompt_v2(
    req: ImgPromptRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None,
        alias="accept",
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes",
    ),
):
    """Image prompt"""
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    if req.input_image is not None:
        req.input_image = base64_to_stream(req.input_image)
    if req.input_mask is not None:
        req.input_mask = base64_to_stream(req.input_mask)

    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for img_prompt in req.image_prompts:
        img_prompt.cn_img = base64_to_stream(img_prompt.cn_img)
        image = ImagePrompt(
            cn_img=img_prompt.cn_img,
            cn_stop=img_prompt.cn_stop,
            cn_weight=img_prompt.cn_weight,
            cn_type=img_prompt.cn_type,
        )
        image_prompts_files.append(image)

    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)

    req.image_prompts = image_prompts_files

    return call_worker(req, accept)
