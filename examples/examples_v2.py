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


upscale_params = {
    "uov_method": "Upscale (Custom)",
    "upscale_value": 3,
    "input_image": ""
}

inpaint_params = {
    "input_image": "",
    "input_mask": None,
    "inpaint_additional_prompt": None,
}

img_prompt_params = {
    "image_prompts": []
}

headers = {
    "accept": "application/json"
}

imgs_base_path = os.path.join(os.path.dirname(__file__), 'imgs')

with open(os.path.join(imgs_base_path, "bear.jpg"), "rb") as f:
    img1 = f.read()
    image_base64 = base64.b64encode(img1).decode('utf-8')
    f.close()

with open(os.path.join(imgs_base_path, "s.jpg"), "rb") as f:
    s = f.read()
    s_base64 = base64.b64encode(s).decode('utf-8')
    f.close()

with open(os.path.join(imgs_base_path, "m.png"), "rb") as f:
    m = f.read()
    m_base64 = base64.b64encode(m).decode('utf-8')
    f.close()


def upscale_vary(image, params=upscale_params) -> dict:
    """
    Upscale or Vary
    """
    params["input_image"] = image
    data = json.dumps(params)
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.img_upscale}",
                             data=data,
                             headers=headers,
                             timeout=300)
    return response.json()


def inpaint_outpaint(input_image: str, input_mask: str = None, params=inpaint_params) -> dict:
    """
    Inpaint or Outpaint
    """
    params["input_image"] = input_image
    params["input_mask"] = input_mask
    params["outpaint_selections"] = ["Left", "Right"]
    params["prompt"] = "cat"
    data = json.dumps(params)
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.inpaint_outpaint}",
                             data=data,
                             headers=headers,
                             timeout=300)
    return response.json()


def image_prompt(img_prompt: list, params: dict) -> dict:
    """
    Image Prompt
    """
    params["image_prompts"] = img_prompt
    data = json.dumps(params)
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.img_prompt}",
                             data=data,
                             headers=headers,
                             timeout=300)
    return response.json()


def image_prompt_with_inpaint(img_prompt: list, input_image: str, input_mask: str, params: dict) -> dict:
    """
    Image Prompt
    """
    params["image_prompts"] = img_prompt
    params["input_image"] = input_image
    params["input_mask"] = input_mask
    params["outpaint_selections"] = ["Left", "Right"]
    data = json.dumps(params)
    response = requests.post(url=f"{cfg.fooocus_host}{cfg.img_prompt}",
                             data=data,
                             headers=headers,
                             timeout=300)
    return response.json()


img_prompt = [
    {
        "cn_img": image_base64,
        "cn_stop": 0.6,
        "cn_weight": 0.6,
        "cn_type": "ImagePrompt"
    }
]
# print(upscale_vary(image=image_base64))
# print(inpaint_outpaint(input_image=s_base64, input_mask=m_base64))
print(image_prompt(img_prompt=img_prompt, params=img_prompt_params))
print(image_prompt_with_inpaint(img_prompt=img_prompt, input_image=s_base64, input_mask=m_base64, params=img_prompt_params))
