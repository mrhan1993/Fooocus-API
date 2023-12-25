import json
import os
import requests
import base64

inpaint_engine = 'v1'

class Config():
    fooocus_host = 'http://127.0.0.1:8888'

    text2img = '/v1/generation/text-to-image'
    img_upscale = '/v2/generation/image-upscale-vary'
    img_upscale1 = '/v1/generation/image-upscale-vary'
    inpaint_outpaint = '/v2/generation/image-inpait-outpaint'
    inpaint_outpaint1 = '/v1/generation/image-inpait-outpaint'
    img_prompt = '/v2/generation/image-prompt'
    img_prompt1 = '/v1/generation/image-prompt'

    job_queue = '/v1/generation/job-queue'
    query_job = '/v1/generation/query-job'

    res_path = '/v1/generation/temp'

cfg = Config()

upscale_params = {
  "prompt": "",
  "negative_prompt": "",
  "style_selections": [
    "Fooocus V2",
    "Fooocus Enhance",
    "Fooocus Sharp"
  ],
  "performance_selection": "Speed",
  "aspect_ratios_selection": "1152*896",
  "image_number": 1,
  "image_seed": -1,
  "sharpness": 2,
  "guidance_scale": 4,
  "base_model_name": "juggernautXL_version6Rundiffusion.safetensors",
  "refiner_model_name": "None",
  "refiner_switch": 0.5,
  "loras": [
    {
      "model_name": "sd_xl_offset_example-lora_1.0.safetensors",
      "weight": 0.1
    }
  ],
  "advanced_params": {
    "disable_preview": False,
    "adm_scaler_positive": 1.5,
    "adm_scaler_negative": 0.8,
    "adm_scaler_end": 0.3,
    "refiner_swap_method": "joint",
    "adaptive_cfg": 7,
    "sampler_name": "dpmpp_2m_sde_gpu",
    "scheduler_name": "karras",
    "overwrite_step": -1,
    "overwrite_switch": -1,
    "overwrite_width": -1,
    "overwrite_height": -1,
    "overwrite_vary_strength": -1,
    "overwrite_upscale_strength": -1,
    "mixing_image_prompt_and_vary_upscale": False,
    "mixing_image_prompt_and_inpaint": False,
    "debugging_cn_preprocessor": False,
    "skipping_cn_preprocessor": False,
    "controlnet_softness": 0.25,
    "canny_low_threshold": 64,
    "canny_high_threshold": 128,
    "freeu_enabled": False,
    "freeu_b1": 1.01,
    "freeu_b2": 1.02,
    "freeu_s1": 0.99,
    "freeu_s2": 0.95,
    "debugging_inpaint_preprocessor": False,
    "inpaint_disable_initial_latent": False,
    "inpaint_engine": inpaint_engine,
    "inpaint_strength": 1,
    "inpaint_respective_field": 1
  },
  "require_base64": False,
  "async_process": False,
  "uov_method": "Upscale (Custom)",
  "upscale_value": 3,
  "input_image": ""
}

inpaint_params = {
  "prompt": "",
  "negative_prompt": "",
  "style_selections": [
    "Fooocus V2",
    "Fooocus Enhance",
    "Fooocus Sharp"
  ],
  "performance_selection": "Speed",
  "aspect_ratios_selection": "1152*896",
  "image_number": 1,
  "image_seed": -1,
  "sharpness": 2,
  "guidance_scale": 4,
  "base_model_name": "juggernautXL_version6Rundiffusion.safetensors",
  "refiner_model_name": "None",
  "refiner_switch": 0.5,
  "loras": [
    {
      "model_name": "sd_xl_offset_example-lora_1.0.safetensors",
      "weight": 0.1
    }
  ],
  "advanced_params": {
    "disable_preview": False,
    "adm_scaler_positive": 1.5,
    "adm_scaler_negative": 0.8,
    "adm_scaler_end": 0.3,
    "refiner_swap_method": "joint",
    "adaptive_cfg": 7,
    "sampler_name": "dpmpp_2m_sde_gpu",
    "scheduler_name": "karras",
    "overwrite_step": -1,
    "overwrite_switch": -1,
    "overwrite_width": -1,
    "overwrite_height": -1,
    "overwrite_vary_strength": -1,
    "overwrite_upscale_strength": -1,
    "mixing_image_prompt_and_vary_upscale": False,
    "mixing_image_prompt_and_inpaint": False,
    "debugging_cn_preprocessor": False,
    "skipping_cn_preprocessor": False,
    "controlnet_softness": 0.25,
    "canny_low_threshold": 64,
    "canny_high_threshold": 128,
    "freeu_enabled": False,
    "freeu_b1": 1.01,
    "freeu_b2": 1.02,
    "freeu_s1": 0.99,
    "freeu_s2": 0.95,
    "debugging_inpaint_preprocessor": False,
    "inpaint_disable_initial_latent": False,
    "inpaint_engine": inpaint_engine,
    "inpaint_strength": 1,
    "inpaint_respective_field": 1
  },
  "require_base64": False,
  "async_process": False,
  "input_image": "",
  "input_mask": None,
  "inpaint_additional_prompt": None,
  "outpaint_selections": [],
  "outpaint_distance_left": 0,
  "outpaint_distance_right": 0,
  "outpaint_distance_top": 0,
  "outpaint_distance_bottom": 0,
}

img_prompt_params = {
  "prompt": "",
  "negative_prompt": "",
  "style_selections": [
    "Fooocus V2",
    "Fooocus Enhance",
    "Fooocus Sharp"
  ],
  "performance_selection": "Speed",
  "aspect_ratios_selection": "1152*896",
  "image_number": 1,
  "image_seed": -1,
  "sharpness": 2,
  "guidance_scale": 4,
  "base_model_name": "juggernautXL_version6Rundiffusion.safetensors",
  "refiner_model_name": "None",
  "refiner_switch": 0.5,
  "loras": [
    {
      "model_name": "sd_xl_offset_example-lora_1.0.safetensors",
      "weight": 0.1
    }
  ],
  "advanced_params": {
    "disable_preview": False,
    "adm_scaler_positive": 1.5,
    "adm_scaler_negative": 0.8,
    "adm_scaler_end": 0.3,
    "refiner_swap_method": "joint",
    "adaptive_cfg": 7,
    "sampler_name": "dpmpp_2m_sde_gpu",
    "scheduler_name": "karras",
    "overwrite_step": -1,
    "overwrite_switch": -1,
    "overwrite_width": -1,
    "overwrite_height": -1,
    "overwrite_vary_strength": -1,
    "overwrite_upscale_strength": -1,
    "mixing_image_prompt_and_vary_upscale": False,
    "mixing_image_prompt_and_inpaint": False,
    "debugging_cn_preprocessor": False,
    "skipping_cn_preprocessor": False,
    "controlnet_softness": 0.25,
    "canny_low_threshold": 64,
    "canny_high_threshold": 128,
    "freeu_enabled": False,
    "freeu_b1": 1.01,
    "freeu_b2": 1.02,
    "freeu_s1": 0.99,
    "freeu_s2": 0.95,
    "debugging_inpaint_preprocessor": False,
    "inpaint_disable_initial_latent": False,
    "inpaint_engine": inpaint_engine,
    "inpaint_strength": 1,
    "inpaint_respective_field": 1
  },
  "require_base64": False,
  "async_process": False,
  "image_prompts": []
}

headers = {
    "accept": "application/json"
}

imgs_base_path = os.path.join(os.path.dirname(__file__), 'imgs')

with open(os.path.join(imgs_base_path, "1485005453352708.jpeg"), "rb") as f:
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


def upscale_vary(image, params = upscale_params) -> dict:
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

def inpaint_outpaint(input_image: str, input_mask: str = None, params = inpaint_params) -> dict:
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
    params["prompt"] = "cat"
    params["image_prompts"] = img_prompt
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
    },{
        "cn_img": s_base64,
        "cn_stop": 0.6,
        "cn_weight": 0.6,
        "cn_type": "ImagePrompt"
    }
]
print(upscale_vary(image=image_base64))
# print(inpaint_outpaint(input_image=s_base64, input_mask=m_base64))
# print(image_prompt(img_prompt=img_prompt, params=img_prompt_params))
