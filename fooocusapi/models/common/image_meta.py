"""
Image meta schema
"""
from typing import List

from fooocus_version import version
from pydantic import BaseModel


class ImageMeta(BaseModel):
    """
    Image meta data model
    """

    metadata_scheme: str = "fooocus"

    base_model: str
    base_model_hash: str

    prompt: str
    full_prompt: List[str]
    prompt_expansion: str

    negative_prompt: str
    full_negative_prompt: List[str]

    performance: str

    style: str

    refiner_model: str = "None"
    refiner_switch: float = 0.5

    loras: List[list]

    resolution: str

    sampler: str = "dpmpp_2m_sde_gpu"
    scheduler: str = "karras"
    seed: str
    adm_guidance: str
    guidance_scale: float
    sharpness: float
    steps: int
    vae_name: str

    version: str = version

    def __repr__(self):
        return ""


def loras_parser(loras: list) -> list:
    """
    Parse lora list
    """
    return [
        [
            lora[0].rsplit('.', maxsplit=1)[:1][0],
            lora[1],
            "hash_not_calculated",
        ] for lora in loras if lora[0] != 'None' and lora[0] is not None]


def image_parse(
        async_tak: object,
        task: dict
) -> dict | str:
    """
    Parse image meta data
    Generate meta data for image from task and async task object
    Args:
        async_tak: async task obj
        task: task obj

    Returns:
        dict: image meta data
    """
    req_param = async_tak.req_param
    meta = ImageMeta(
        metadata_scheme=req_param.meta_scheme,
        base_model=req_param.base_model_name.rsplit('.', maxsplit=1)[:1][0],
        base_model_hash='',
        prompt=req_param.prompt,
        full_prompt=task['positive'],
        prompt_expansion=task['expansion'],
        negative_prompt=req_param.negative_prompt,
        full_negative_prompt=task['negative'],
        performance=req_param.performance_selection,
        style=str(req_param.style_selections),
        refiner_model=req_param.refiner_model_name,
        refiner_switch=req_param.refiner_switch,
        loras=loras_parser(req_param.loras),
        resolution=str(tuple([int(n) for n in req_param.aspect_ratios_selection.split('*')])),
        sampler=req_param.advanced_params.sampler_name,
        scheduler=req_param.advanced_params.scheduler_name,
        seed=str(task['task_seed']),
        adm_guidance=str((
            req_param.advanced_params.adm_scaler_positive,
            req_param.advanced_params.adm_scaler_negative,
            req_param.advanced_params.adm_scaler_end)),
        guidance_scale=req_param.guidance_scale,
        sharpness=req_param.sharpness,
        steps=-1,
        vae_name=req_param.advanced_params.vae_name,
        version=version
    )
    if meta.metadata_scheme not in ["fooocus", "a111"]:
        meta.metadata_scheme = "fooocus"
    if meta.metadata_scheme == "fooocus":
        meta_dict = meta.model_dump()
        for i, lora in enumerate(meta.loras):
            attr_name = f"lora_combined_{i+1}"
            lr = [str(x) for x in lora]
            meta_dict[attr_name] = f"{lr[0]} : {lr[1]}"
    else:
        meta_dict = meta.model_dump()
    return meta_dict
