"""
Examples codes for Fooocus API
"""
import json
import os
import base64
import requests


class Config:
    """
    Config
    Attributes:
        fooocus_host (str): Fooocus API host
        text2img_ip (str): Text to Image with IP
        img_upscale (str): Upscale or Vary
        inpaint_outpaint (str): Inpaint or Outpaint
        img_prompt (str): Image Prompt
    """
    fooocus_host = 'http://127.0.0.1:8888'

    text2img_ip = '/v2/generation/text-to-image-with-ip'
    img_upscale = '/v2/generation/image-upscale-vary'
    inpaint_outpaint = '/v2/generation/image-inpaint-outpaint'
    img_prompt = '/v2/generation/image-prompt'


def read_image(image_name: str) -> str:
    """
    Read image from file
    Args:
        image_name (str): Image file name
    Returns:
        str: Image base64
    """
    path = os.path.join('imgs', image_name)
    with open(path, "rb") as f:
        image = f.read()
        f.close()
    return base64.b64encode(image).decode('utf-8')


class ImageList:
    """
    Image List
    """
    bear = read_image('bear.jpg')
    image_prompt_0 = read_image('image_prompt-0.jpg')
    image_prompt_1 = read_image('image_prompt-1.png')
    image_prompt_2 = read_image('image_prompt-2.png')
    image_prompt_3 = read_image('image_prompt-3.png')
    inpaint_source = read_image('inpaint_source.jpg')
    inpaint_mask = read_image('inpaint_mask.png')
    source_face_f = read_image('source_face_female.png')
    source_face_m = read_image('source_face_man.png')
    target_face = read_image('target_face.png')


def upscale_vary(params: dict) -> dict:
    """
    Upscale or Vary
    Args:
        params (dict): Params
    Returns:
        dict: Response
    """
    data = json.dumps(params)
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.img_upscale}",
        data=data,
        timeout=300)
    return response.json()


def inpaint_outpaint(params: dict = None) -> dict:
    """
    Inpaint or Outpaint
    Args:
        params (dict): Params
    Returns:
        dict: Response
    """
    data = json.dumps(params)
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.inpaint_outpaint}",
        data=data,
        timeout=300)
    return response.json()


def image_prompt(params: dict) -> dict:
    """
    Image Prompt
    Args:
        params (dict): Params
    Returns:
        dict: Response
    """
    data = json.dumps(params)
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.img_prompt}",
        data=data,
        timeout=300)
    return response.json()


def text2image_image_prompt(params: dict) -> dict:
    """
    Text to image with image Prompt
    Args:
        params (dict): Params
    Returns:
        dict: Response
    """
    params["outpaint_selections"] = ["Left", "Right"]
    data = json.dumps(params)
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.text2img_ip}",
        data=data,
        timeout=300)
    return response.json()


# ################################################################
# Upscale or Vary
# ################################################################

# Upscale (2x) example
uov_params = {
    "input_image": ImageList.image_prompt_0,
    "performance_selection": "Lightning",
    "uov_method": "Upscale (2x)",
    "async_process": True
}

upscale_result = upscale_vary(params=uov_params)
print(
    json.dumps(
        upscale_result,
        indent=4,
        ensure_ascii=False
    ))

# Vary (Strong) example
uov_params['uov_method'] = 'Vary (Strong)'

vary_result = upscale_vary(params=uov_params)
print(
    json.dumps(
        vary_result,
        indent=4,
        ensure_ascii=False
    ))


# ################################################################
# Inpaint or Outpaint
# ################################################################

# Inpaint outpaint example
inpaint_params = {
    "prompt": "a cat",  # use background prompt to remove anything what you don't want
    "performance_selection": "Speed",  # use Lightning the quality is not good
    "input_image": ImageList.inpaint_source,
    "input_mask": ImageList.inpaint_mask,
    "outpaint_selections": ["Left", "Right"],
    "async_process": False
}

inpaint_result = inpaint_outpaint(params=inpaint_params)
print(json.dumps(inpaint_result))


# ################################################################
# Image Prompt example
# more detail for image prompt can be found:
# https://github.com/lllyasviel/Fooocus/discussions/557
# ################################################################

# face swap example
# This parameter comes from the default parameter of the Fooocus interface,
# but the effect of using this parameter for face swap with Fooocus is general.
# It is too large for the return of the original image. If necessary, try to adjust more parameters.
face_swap_params = {
    "performance_selection": "Speed",
    "aspect_ratios_selection": "896*1152",
    "image_prompts": [
        {
            "cn_img": ImageList.source_face_m,
            "cn_stop": 0.5,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        }, {
            "cn_img": ImageList.target_face,
            "cn_stop": 0.9,
            "cn_weight": 0.75,
            "cn_type": "FaceSwap"
        }
    ],
    "async_process": False
}

face_swap_result = image_prompt(params=face_swap_params)
print(json.dumps(face_swap_result))


# ################################################################
# Text to image with image Prompt
# ################################################################

# Text to image with image Prompt example
t2i_ip_params = {
    "prompt": "a cat",
    "performance_selection": "Speed",
    "image_prompts": [
        {
            "cn_img": ImageList.image_prompt_1,
            "cn_stop": 0.6,
            "cn_weight": 0.8,
            "cn_type": "ImagePrompt"
        }, {
            "cn_img": ImageList.image_prompt_2,
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        }
    ],
    "async_process": False
}

t2i_ip_result = text2image_image_prompt(params=t2i_ip_params)
print(json.dumps(t2i_ip_result))

# ################################################################
# Image Enhance
# ################################################################

# Image Enhance

import requests
import json

url = "http://localhost:8888/v2/generation/image-enhance"

headers = {
    "Content-Type": "application/json"
}

data = {
    "enhance_input_image": "https://raw.githubusercontent.com/mrhan1993/Fooocus-API/main/examples/imgs/source_face_man.png",
    "enhance_checkbox": True,
    "enhance_uov_method": "Vary (Strong)",
    "enhance_uov_processing_order": "Before First Enhancement",
    "enhance_uov_prompt_type": "Original Prompts",
    "enhance_ctrlnets": [
        {
            "enhance_enabled": True,
            "enhance_mask_dino_prompt": "face",
            "enhance_prompt": "",
            "enhance_negative_prompt": "",
            "enhance_mask_model": "sam",
            "enhance_mask_cloth_category": "full",
            "enhance_mask_sam_model": "vit_b",
            "enhance_mask_text_threshold": 0.25,
            "enhance_mask_box_threshold": 0.3,
            "enhance_mask_sam_max_detections": 0,
            "enhance_inpaint_disable_initial_latent": False,
            "enhance_inpaint_engine": "v2.6",
            "enhance_inpaint_strength": 1.0,
            "enhance_inpaint_respective_field": 0.618,
            "enhance_inpaint_erode_or_dilate": 0.0,
            "enhance_mask_invert": False
        }
    ]
}

response = requests.post(
    url,
    headers=headers,
    data=json.dumps(data),
    timeout=180)

if response.status_code == 200:
    print("Request successful!")
    print("Response:", response.json())
else:
    print("Request failed with status code:", response.status_code)
    print("Response:", response.text)
