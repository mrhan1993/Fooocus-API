import json
import requests
import os
import base64

from models_v2 import *

class Config():
    fooocus_host = 'http://127.0.0.1:8888'

    text2img = '/v1/generation/text-to-image'
    img_upscale = '/v2/generation/image-upscale-vary'
    inpaint_outpaint = '/v2/generation/image-inpait-outpaint'
    img_prompt = '/v2/generation/image-prompt'

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

def upscales(params: ImgUpscaleOrVaryParams) -> dict:
    """
    image-upscale-vary
    """
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.img_upscale}",
                            data=json.dumps(params.model_dump()),
                            timeout=30)
    return response.json()


def in_out_paint(params: ImgInpaintOrOutpaintParams,) -> dict:
    """
    inpaint or outpaint, move watermark, replace sth, extend image
    """
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.inpaint_outpaint}",
                            data=json.dumps(params.model_dump()),
                            timeout=30)
    return response.json()


def image_prompt(params: ImgPromptParams = ImgPromptParams()) -> dict:
    """
    image to image
    """
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.img_prompt}",
                            data=json.dumps(params.model_dump()),
                            timeout=30)
    return response.json()


imgs_base_path = os.path.join(os.path.dirname(__file__), 'imgs')

# example for text2img
params = {
    "prompt": "sunshine, river, trees"
}
txt2img(Text2ImgParams(**params))


# example for image-upscale-vary
image = open(os.path.join(imgs_base_path,'1485005453352708.jpeg'), 'rb').read()
params = {
    "input_image": base64.b64encode(image).decode('utf-8'),
    "uov_method": UpscaleOrVaryMethod.upscale_15
}
upscales(ImgUpscaleOrVaryParams(**params))


# example for image-inpait-outpaint
# inpaint
params = {
    "prompt": "background",
    "input_image": base64.b64encode(open(os.path.join(imgs_base_path,'s.jpg'), 'rb').read()).decode('utf-8'),
    "input_mask": base64.b64encode(open(os.path.join(imgs_base_path,'m.png'), 'rb').read()).decode('utf-8')
}
in_out_paint(ImgInpaintOrOutpaintParams(**params))

# outpaint
params = {
    "prompt": "",
    "input_image": base64.b64encode(open(os.path.join(imgs_base_path,'s.jpg'), 'rb').read()).decode('utf-8'),
    "outpaint_selections": ["Left", "Right"],
    "outpaint_distance_left": 200,
    "outpaint_distance_right": 50
}
in_out_paint(ImgInpaintOrOutpaintParams(**params))


# # example for image-prompt
params = {
    "prompt": "a tiger on the ground",
    "image_prompts": [
        {"cn_img": base64.b64encode(open(os.path.join(imgs_base_path,'1485005453352708.jpeg'), 'rb').read()).decode('utf-8'),
         "cn_stop": 0.5,
         "cn_weight": 0.5,
         "cn_type": ControlNetEnum.imagePrompt},
         {"cn_img": base64.b64encode(open(os.path.join(imgs_base_path,'s.jpg'), 'rb').read()).decode('utf-8'),
         "cn_stop": 0.5,
         "cn_weight": 0.5,
         "cn_type": ControlNetEnum.imagePrompt}
    ]
}
image_prompt(ImgPromptParams(**params))


# you can also instantiate the params object like the example below.
params = ImgUpscaleOrVaryParams(
    input_image=base64.b64encode(image).decode('utf-8'),
    uov_method=UpscaleOrVaryMethod.upscale_15
)
upscales(params)