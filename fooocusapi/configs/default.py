"""
Static variables for Fooocus API
"""
img_generate_responses = {
    "200": {
        "description": "PNG bytes if request's 'Accept' header is 'image/png', otherwise JSON",
        "content": {
            "application/json": {
                "example": [{
                        "base64": "...very long string...",
                        "seed": "1050625087",
                        "finish_reason": "SUCCESS",
                    }]
            },
            "application/json async": {
                "example": {
                    "job_id": 1,
                    "job_type": "Text to Image"
                }
            },
            "image/png": {
                "example": "PNG bytes, what did you expect?"
            },
        },
    }
}

default_inpaint_engine_version = "v2.6"

default_styles = ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"]
default_base_model_name = "juggernautXL_v8Rundiffusion.safetensors"
default_refiner_model_name = "None"
default_refiner_switch = 0.5
default_loras = [[True, "sd_xl_offset_example-lora_1.0.safetensors", 0.1]]
default_cfg_scale = 7.0
default_prompt_negative = ""
default_aspect_ratio = "1152*896"
default_sampler = "dpmpp_2m_sde_gpu"
default_scheduler = "karras"

available_aspect_ratios = [
    "704*1408",
    "704*1344",
    "768*1344",
    "768*1280",
    "832*1216",
    "832*1152",
    "896*1152",
    "896*1088",
    "960*1088",
    "960*1024",
    "1024*1024",
    "1024*960",
    "1088*960",
    "1088*896",
    "1152*896",
    "1152*832",
    "1216*832",
    "1280*768",
    "1344*768",
    "1344*704",
    "1408*704",
    "1472*704",
    "1536*640",
    "1600*640",
    "1664*576",
    "1728*576",
]

uov_methods = [
    "Disabled",
    "Vary (Subtle)",
    "Vary (Strong)",
    "Upscale (1.5x)",
    "Upscale (2x)",
    "Upscale (Fast 2x)",
    "Upscale (Custom)",
]

outpaint_expansions = ["Left", "Right", "Top", "Bottom"]


def get_aspect_ratio_value(label: str) -> str:
    """
    Get aspect ratio
    Args:
        label: str, aspect ratio

    Returns:

    """
    return label.split(" ")[0].replace("×", "*")
