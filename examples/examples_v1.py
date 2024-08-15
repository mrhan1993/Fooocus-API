"""
Examples codes for Fooocus API
"""
import json
import os
import requests


class Config:
    """
    Config
    Attributes:
        fooocus_host (str): Fooocus API host
        img_upscale (str): Upscale or Vary
        inpaint_outpaint (str): Inpaint or Outpaint
        img_prompt (str): Image Prompt
    """
    fooocus_host = 'http://127.0.0.1:8888'

    text2image = '/v1/generation/text-to-image'
    img_upscale = '/v1/generation/image-upscale-vary'
    inpaint_outpaint = '/v1/generation/image-inpaint-outpaint'
    img_prompt = '/v1/generation/image-prompt'


def read_image(image_name: str) -> bytes:
    """
    Read image from file
    Args:
        image_name (str): Image file name
    Returns:
        image (bytes): Image data
    """
    path = os.path.join('imgs', image_name)
    with open(path, "rb") as f:
        image = f.read()
        f.close()
    return image


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


def text2image(params: dict) -> dict:
    """
    Text to image
    Args:
        params (dict): Params
    Returns:
        dict: Response
    """
    data = json.dumps(params)
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.text2image}",
        data=data,
        timeout=300)
    return response.json()


def upscale_vary(image: bytes, params: dict) -> dict:
    """
    Upscale or Vary
    Args:
        image (bytes): Image data
        params (dict): Params
    Returns:
        dict: Response
    """
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.img_upscale}",
        data=params,
        files={
            'input_image': image,
        },
        timeout=300)
    return response.json()


def inpaint_outpaint(
        params: dict,
        input_image: bytes,
        input_mask: bytes = None) -> dict:
    """
    Inpaint or Outpaint
    Args:
        params (dict): Params
        input_image (bytes): Image data
        input_mask (bytes): Image mask data
    Returns:
        dict: Response
    """
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.inpaint_outpaint}",
        data=params,
        files={
            'input_image': input_image,
            'input_mask': input_mask
        },
        timeout=300)
    return response.json()


def image_prompt(
        params: dict,
        input_image: bytes = None,
        input_mask: bytes = None,
        cn_img1: bytes = None,
        cn_img2: bytes = None,
        cn_img3: bytes = None,
        cn_img4: bytes = None,) -> dict:
    """
    Image Prompt
    Args:
        params (dict): Params
        input_image (bytes): Image data
        input_mask (bytes): Image mask data
        cn_img1 (bytes): Image data
        cn_img2 (bytes): Image data
        cn_img3 (bytes): Image data
        cn_img4 (bytes): Image data
    Returns:
        dict: Response
    """
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.img_prompt}",
        data=params,
        files={
            'input_image': input_image,
            'input_mask': input_mask,
            'cn_img1': cn_img1,
            'cn_img2': cn_img2,
            'cn_img3': cn_img3,
            'cn_img4': cn_img4
        },
        timeout=300)
    return response.json()


# ###############################################################
# Text to image example
# ################################################################

# Text to image example
t2i_params = {
    "prompt": "a cat",
    "performance_selection": "Lightning",
    "aspect_ratios_selection": "896*1152",
    "async_process": True
}

t2i_result = text2image(params=t2i_params)
print(json.dumps(t2i_result))


# ###############################################################
# Upscale or Vary example
# ################################################################

# Upscale or Vary example
up_params = {
    "uov_method": "Upscale (2x)",
    "async_process": True
}

up_result = upscale_vary(
    image=ImageList.bear,
    params=up_params
)
print(json.dumps(up_result))


# ###############################################################
# Inpaint or Outpaint example
# ################################################################

# Inpaint or Outpaint example
io_params = {
    "prompt": "a cat",
    "outpaint_selections": "Left,Top",
    "async_process": True
}

io_result = inpaint_outpaint(
    params=io_params,
    input_image=ImageList.inpaint_source,
    input_mask=ImageList.inpaint_mask
)
print(json.dumps(io_result))


# ###############################################################
# Image prompt example
# ################################################################

# Image prompt example
ip_params = {
    "prompt": "a cat",
    "image_prompts": [
        {
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        },
        {
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        }
    ]
}

ip_result = image_prompt(
    params=ip_params,
    cn_img1=ImageList.image_prompt_0,
    cn_img2=ImageList.image_prompt_1,
)
print(json.dumps(ip_result))

# ###############################################################
# Image Enhance
# ################################################################

# Image Enhance

import requests

url = "http://localhost:8888/v1/generation/image-enhance"

# Define the file path and other form data
file_path = "./examples/imgs/source_face_man.png"
form_data = {
    "enhance_checkbox": True,
    "enhance_uov_method": "Disabled",
    "enhance_enabled_1": True,
    "enhance_mask_dino_prompt_1": "face",
    "enhance_enabled_2": True,
    "enhance_mask_dino_prompt_2": "eyes",
}

# Open the file and prepare it for the request
with open(file_path, "rb") as f:
    image = f.read()
    f.close()

# Send the request
response = requests.post(
    url,
    files={"enhance_input_image": image},
    data=form_data,
    timeout=180)

# Print the response content
print(response.text)
