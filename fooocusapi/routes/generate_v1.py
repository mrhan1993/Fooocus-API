"""Generate API v1"""
from typing import List, Optional
from modules.util import HWC3
from fastapi import (
    UploadFile,
    APIRouter,
    Depends,
    Header,
    Query,
    File
)
from fooocusapi.models.models import (
    ImgInpaintOrOutpaintRequest,
    ImgUpscaleOrVaryRequest,
    DescribeImageResponse,
    GeneratedImageResult,
    DescribeImageType,
    AsyncJobResponse,
    ImgPromptRequest,
    Text2ImgRequest
)

from fooocusapi.utils.api_utils import (
    img_generate_responses,
    api_key_auth,
    call_worker
)
from fooocusapi.utils.img_utils import read_input_image


secure_router = APIRouter(dependencies=[Depends(api_key_auth)])


@secure_router.post(
    "/v1/generation/text-to-image",
    response_model=List[GeneratedImageResult] | AsyncJobResponse,
    responses=img_generate_responses,
)
def text2img_generation(
    req: Text2ImgRequest,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None,
        alias="accept",
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes",
    ),
):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post(
    "/v1/generation/image-upscale-vary",
    response_model=List[GeneratedImageResult] | AsyncJobResponse,
    responses=img_generate_responses,
)
def img_upscale_or_vary(
    input_image: UploadFile,
    req: ImgUpscaleOrVaryRequest = Depends(ImgUpscaleOrVaryRequest.as_form),
    accept: str = Header(None),
    accept_query: str | None = Query(
        None,
        alias="accept",
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes",
    ),
):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post(
    "/v1/generation/image-inpaint-outpaint",
    response_model=List[GeneratedImageResult] | AsyncJobResponse,
    responses=img_generate_responses,
)
def img_inpaint_or_outpaint(
    input_image: UploadFile,
    req: ImgInpaintOrOutpaintRequest = Depends(ImgInpaintOrOutpaintRequest.as_form),
    accept: str = Header(None),
    accept_query: str | None = Query(
        None,
        alias="accept",
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes",
    ),
):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post(
    "/v1/generation/image-prompt",
    response_model=List[GeneratedImageResult] | AsyncJobResponse,
    responses=img_generate_responses,
)
def img_prompt(
    cn_img1: Optional[UploadFile] = File(None),
    req: ImgPromptRequest = Depends(ImgPromptRequest.as_form),
    accept: str = Header(None),
    accept_query: str | None = Query(
        None,
        alias="accept",
        description="Parameter to overvide 'Accept' header, 'image/png' for output bytes",
    ),
):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post("/v1/tools/describe-image", response_model=DescribeImageResponse)
def describe_image(
    image: UploadFile,
    task_type: DescribeImageType = Query(
        DescribeImageType.photo, description="Image type, 'Photo' or 'Anime'"
    ),
):
    """
    Describe image, return text description of image.
    Args:
        image: Image file to be described.
        task_type: Image type, 'Photo' or 'Anime'
    """
    if task_type == DescribeImageType.photo:
        from extras.interrogate import (
            default_interrogator as default_interrogator_photo,
        )

        interrogator = default_interrogator_photo
    else:
        from extras.wd14tagger import default_interrogator as default_interrogator_anime

        interrogator = default_interrogator_anime
    img = HWC3(read_input_image(image))
    result = interrogator(img)
    return DescribeImageResponse(describe=result)
