import json
import requests
import os

from models_v1 import *

class Config():
    fooocus_host = 'http://127.0.0.1:8888'

    text2img = '/v1/generation/text-to-image'
    img_upscale = '/v1/generation/image-upscale-vary'
    inpaint_outpaint = '/v1/generation/image-inpait-outpaint'
    img_prompt = '/v1/generation/image-prompt'

    job_queue = '/v1/generation/job-queue'
    query_job = '/v1/generation/query-job'

cfg = Config()


def txt2img(params: Text2ImgParams) -> dict:
    """
    text to image
    """
    date = json.dumps(params.model_dump())
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.text2img}",
                        data=date,
                        timeout=30)
    return response.json()


def upscales(image: bytes,
             params: ImgUpscaleOrVaryParams = ImgUpscaleOrVaryParams()) -> dict:
    """
    image-upscale-vary
    """
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.img_upscale}",
                            data=json.loads(json.dumps(params.model_dump())),
                            files={"input_image": image},
                            timeout=30)
    return response.json()


def in_out_paint(image: bytes,
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


def image_prompt(cn_img1: bytes,
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


imgs_base_path = os.path.join(os.path.dirname(__file__), 'imgs')

# example for text2img
params = {
    "prompt": "sunshine, river, trees"
}
txt2img(Text2ImgParams(**params))


# exaple for upscales
params = {
    "uov_method": "Upscale (2x)"
}
upscales(image=open(os.path.join(imgs_base_path, "1485005453352708.jpeg"), "rb").read(),
         params=ImgUpscaleOrVaryParams(**params))


# example for in_out_paint
# inpaint
params = {
    "prompt": "a cat"
}
in_out_paint(image=open(os.path.join(imgs_base_path, "s.jpg"), "rb").read(),
             params=ImgInpaintOrOutpaintParams(**params),
             mask=open(os.path.join(imgs_base_path, "m.png"), "rb").read())

# outpaint
params = {
    "prompt": "",
    "outpaint_selections": "Left,Right",
    "outpaint_distance_left": 200,
    "outpaint_distance_right": 50
}
in_out_paint(image=open(os.path.join(imgs_base_path, "1485005453352708.jpeg"), "rb").read(),
             params=ImgInpaintOrOutpaintParams(**params))


# example for image_prompt
params = {
    "cn_stop1": 0.5,
    "cn_weight1": 0.5,
    "cn_type1": "ImagePrompt",
    "cn_stop2": 0.5,
    "cn_weight2": 0.5,
    "cn_type2": "ImagePrompt"
}
image_prompt(cn_img1=open(os.path.join(imgs_base_path, "1485005453352708.jpeg"), "rb").read(),
             cn_img2=open(os.path.join(imgs_base_path, "s.jpg"), "rb").read(),
             params=ImagePromptParams(**params))

# you can also instantiate the params object like the example below.
params = ImgUpscaleOrVaryParams(
    uov_method=UpscaleOrVaryMethod.upscale_20
)
upscales(image=open(os.path.join(imgs_base_path, "1485005453352708.jpeg"), "rb").read(),
         params=params)
