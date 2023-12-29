import json
import requests
import os
import base64
from examples.models import ControlNetEnum, ImagePromptParams, ImagePromptParamsJson, ImgInpaintOrOutpaintParams, ImgInpaintOrOutpaintParamsJson, ImgUpscaleOrVaryParams, ImgUpscaleOrVaryParamsJson, Text2ImgParams, UpscaleOrVaryMethod

from models import *

class Config():
    fooocus_host = 'http://127.0.0.1:8888'

    text2img = '/v1/generation/text-to-image'
    img_upscale = '/v1/generation/image-upscale-vary'
    inpaint_outpaint = '/v1/generation/image-inpait-outpaint'
    img_prompt = '/v1/generation/image-prompt'

    img_upscale_v2 = '/v2/generation/image-upscale-vary'
    inpaint_outpaint_v2 = '/v2/generation/image-inpait-outpaint'
    img_prompt_v2 = '/v2/generation/image-prompt'

cfg = Config()


def txt2img(params: Text2ImgParams) -> dict:
    """
    text to image
    """
    data = json.dumps(params.model_dump())
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.text2img}",
                        data=data,
                        timeout=30)
    return response.json()


def upscale_vary_v1(image: bytes,
             params: ImgUpscaleOrVaryParams = ImgUpscaleOrVaryParams()) -> dict:
    """
    image-upscale-vary
    """
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.img_upscale}",
                            data=json.loads(json.dumps(params.model_dump())),
                            files={"input_image": image},
                            timeout=30)
    return response.json()


def upscale_vary_v2(params: ImgUpscaleOrVaryParamsJson) -> dict:
    """
    image-upscale-vary
    """
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.img_upscale_v2}",
                            data=json.dumps(params.model_dump()),
                            timeout=30)
    return response.json()


def inpaint_outpaint_v1(image: bytes,
                 params: ImgInpaintOrOutpaintParams,
                 mask: bytes = None,
                 ) -> dict:
    """
    inpaint or outpaint, move watermark, replace sth, extend image
    """
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.inpaint_outpaint}",
                            data=json.loads(json.dumps(params.model_dump())),
                            files={"input_image": image,
                                   "input_mask": mask},
                            timeout=30)
    return response.json()


def inpaint_outpaint_v2(params: ImgInpaintOrOutpaintParamsJson,) -> dict:
    """
    inpaint or outpaint, move watermark, replace sth, extend image
    """
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.inpaint_outpaint_v2}",
                            data=json.dumps(params.model_dump()),
                            timeout=30)
    return response.json()


def image_prompt_v1(cn_img1: bytes,
                    cn_img2: bytes = None,
                    cn_img3: bytes = None,
                    cn_img4: bytes = None,
                    params: ImagePromptParams = ImagePromptParams()) -> dict:
    """
    image to image
    """
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.img_prompt}",
                            data=json.loads(json.dumps(params.model_dump())),
                            files={
                                "cn_img1": cn_img1,
                                "cn_img2": cn_img2,
                                "cn_img3": cn_img3,
                                "cn_img4": cn_img4
                            },
                            timeout=30)
    return response.json()


def image_prompt_v2(params: ImagePromptParamsJson = ImagePromptParamsJson()) -> dict:
    """
    image to image
    """
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.img_prompt_v2}",
                            data=json.dumps(params.model_dump()),
                            timeout=30)
    return response.json()


if __name__ == '__main__':
    imgs_base_path = os.path.join(os.path.dirname(__file__), 'imgs')

    input_image = open(os.path.join(imgs_base_path,'bear.jpg'), 'rb').read()
    input_source = open(os.path.join(imgs_base_path,'s.jpg'), 'rb').read()
    input_mask = open(os.path.join(imgs_base_path,'m.png'), 'rb').read()

    # example for text2img
    params = {
        "prompt": "sunshine, river, trees"
    }
    res = txt2img(Text2ImgParams(**params))
    print(res)


    # exaple for upscales
    params = {
        "uov_method": "Upscale (Custom)",
        "upscale_value": 2.0
    }
    res = upscale_vary_v1(image=input_image,
            params=ImgUpscaleOrVaryParams(**params))
    print(res)

    # use v2
    params["input_image"] = base64.b64encode(input_image).decode('utf-8')
    res = upscale_vary_v2(params=ImgUpscaleOrVaryParamsJson(**params))
    print(res)


    # example for in_out_paint
    # inpaint
    params = {
        "prompt": "a cat"
    }
    res = inpaint_outpaint_v1(image=input_source,
                params=ImgInpaintOrOutpaintParams(**params),
                mask=input_mask)
    print(res)

    # use v2
    params["input_image"] = base64.b64encode(input_source).decode('utf-8')
    params["input_mask"] = base64.b64encode(input_mask).decode('utf-8')
    res = inpaint_outpaint_v2(params=ImgInpaintOrOutpaintParamsJson(**params))
    print(res)


    # outpaint
    params = {
        "prompt": "",
        "outpaint_selections": "Left,Right",
        "outpaint_distance_left": 200,
        "outpaint_distance_right": 50
    }
    res = inpaint_outpaint_v1(image=input_image,
                params=ImgInpaintOrOutpaintParams(**params))
    print(res)

    # use v2
    params["input_image"] = base64.b64encode(input_image).decode('utf-8')
    params["outpaint_selections"] = ["Left", "Right"]
    res = inpaint_outpaint_v2(params=ImgInpaintOrOutpaintParamsJson(**params))
    print(res)


    # example for image_prompt
    params = {
        "cn_stop1": 0.5,
        "cn_weight1": 0.5,
        "cn_type1": "ImagePrompt",
        "cn_stop2": 0.5,
        "cn_weight2": 0.5,
        "cn_type2": "ImagePrompt"
    }
    res = image_prompt_v1(cn_img1=input_image,
                cn_img2=input_source,
                params=ImagePromptParams(**params))
    print(res)

    # use v2
    params = {
        "prompt": "a tiger on the ground",
        "image_prompts": [
            {"cn_img": base64.b64encode(input_image).decode('utf-8'),
            "cn_stop": 0.5,
            "cn_weight": 0.5,
            "cn_type": ControlNetEnum.imagePrompt},
            {"cn_img": base64.b64encode(input_source).decode('utf-8'),
            "cn_stop": 0.5,
            "cn_weight": 0.5,
            "cn_type": ControlNetEnum.imagePrompt}
        ]
    }
    res = image_prompt_v2(params=ImagePromptParamsJson(**params))
    print(res)


    # you can also instantiate the params object like the example below.
    params = ImgUpscaleOrVaryParams(
        uov_method=UpscaleOrVaryMethod.upscale_20
    )
    res = upscale_vary_v1(image=input_image,
            params=params)
    print(res)
