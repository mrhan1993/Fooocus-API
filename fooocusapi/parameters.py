from typing import Dict, List, Tuple
import numpy as np
import copy

from pydantic import BaseModel, Field


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
default_loras = [["sd_xl_offset_example-lora_1.0.safetensors", 0.1]]
default_cfg_scale = 4.0
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


class AdvancedParams(BaseModel):
    """Common params object AdvancedParams"""
    disable_preview: bool = Field(False, description="Disable preview during generation")
    disable_intermediate_results: bool = Field(False, description="Disable intermediate results")
    disable_seed_increment: bool = Field(False, description="Disable Seed Increment")
    adm_scaler_positive: float = Field(1.5, description="Positive ADM Guidance Scaler", ge=0.1, le=3.0)
    adm_scaler_negative: float = Field(0.8, description="Negative ADM Guidance Scaler", ge=0.1, le=3.0)
    adm_scaler_end: float = Field(0.3, description="ADM Guidance End At Step", ge=0.0, le=1.0)
    adaptive_cfg: float = Field(7.0, description="CFG Mimicking from TSNR", ge=1.0, le=30.0)
    sampler_name: str = Field(default_sampler, description="Sampler")
    scheduler_name: str = Field(default_scheduler, description="Scheduler")
    overwrite_step: int = Field(-1, description="Forced Overwrite of Sampling Step", ge=-1, le=200)
    overwrite_switch: float = Field(-1, description="Forced Overwrite of Refiner Switch Step", ge=-1, le=1)
    overwrite_width: int = Field(-1, description="Forced Overwrite of Generating Width", ge=-1, le=2048)
    overwrite_height: int = Field(-1, description="Forced Overwrite of Generating Height", ge=-1, le=2048)
    overwrite_vary_strength: float = Field(-1, description='Forced Overwrite of Denoising Strength of "Vary"', ge=-1, le=1.0)
    overwrite_upscale_strength: float = Field(-1, description='Forced Overwrite of Denoising Strength of "Upscale"', ge=-1, le=1.0)
    mixing_image_prompt_and_vary_upscale: bool = Field(False, description="Mixing Image Prompt and Vary/Upscale")
    mixing_image_prompt_and_inpaint: bool = Field(False, description="Mixing Image Prompt and Inpaint")
    debugging_cn_preprocessor: bool = Field(False, description="Debug Preprocessors")
    skipping_cn_preprocessor: bool = Field(False, description="Skip Preprocessors")
    canny_low_threshold: int = Field(64, description="Canny Low Threshold", ge=1, le=255)
    canny_high_threshold: int = Field(128, description="Canny High Threshold", ge=1, le=255)
    refiner_swap_method: str = Field('joint', description="Refiner swap method")
    controlnet_softness: float = Field(0.25, description="Softness of ControlNet", ge=0.0, le=1.0)
    freeu_enabled: bool = Field(False, description="FreeU enabled")
    freeu_b1: float = Field(1.01, description="FreeU B1")
    freeu_b2: float = Field(1.02, description="FreeU B2")
    freeu_s1: float = Field(0.99, description="FreeU B3")
    freeu_s2: float = Field(0.95, description="FreeU B4")
    debugging_inpaint_preprocessor: bool = Field(False, description="Debug Inpaint Preprocessing")
    inpaint_disable_initial_latent: bool = Field(False, description="Disable initial latent in inpaint")
    inpaint_engine: str = Field('v2.6', description="Inpaint Engine")
    inpaint_strength: float = Field(1.0, description="Inpaint Denoising Strength", ge=0.0, le=1.0)
    inpaint_respective_field: float = Field(1.0, description="Inpaint Respective Field", ge=0.0, le=1.0)
    inpaint_mask_upload_checkbox: bool = Field(False, description="Upload Mask")
    invert_mask_checkbox: bool = Field(False, description="Invert Mask")
    inpaint_erode_or_dilate: int = Field(0, description="Mask Erode or Dilate", ge=-64, le=64)


class ImageGenerationParams:
    def __init__(
        self,
        prompt: str,
        negative_prompt: str,
        style_selections: List[str],
        performance_selection: str,
        aspect_ratios_selection: str,
        image_number: int,
        image_seed: int | None,
        sharpness: float,
        guidance_scale: float,
        base_model_name: str,
        refiner_model_name: str,
        refiner_switch: float,
        loras: List[Tuple[str, float]],
        uov_input_image: np.ndarray | None,
        uov_method: str,
        upscale_value: float | None,
        outpaint_selections: List[str],
        outpaint_distance_left: int,
        outpaint_distance_right: int,
        outpaint_distance_top: int,
        outpaint_distance_bottom: int,
        inpaint_input_image: Dict[str, np.ndarray] | None,
        inpaint_additional_prompt: str | None,
        image_prompts: List[Tuple[np.ndarray, float, float, str]],
        advanced_params: List[any] | None,
        save_extension: str,
        require_base64: bool,
    ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.style_selections = style_selections
        self.performance_selection = performance_selection
        self.aspect_ratios_selection = aspect_ratios_selection
        self.image_number = image_number
        self.image_seed = image_seed
        self.sharpness = sharpness
        self.guidance_scale = guidance_scale
        self.base_model_name = base_model_name
        self.refiner_model_name = refiner_model_name
        self.refiner_switch = refiner_switch
        self.loras = loras
        self.uov_input_image = uov_input_image
        self.uov_method = uov_method
        self.upscale_value = upscale_value
        self.outpaint_selections = outpaint_selections
        self.outpaint_distance_left = outpaint_distance_left
        self.outpaint_distance_right = outpaint_distance_right
        self.outpaint_distance_top = outpaint_distance_top
        self.outpaint_distance_bottom = outpaint_distance_bottom
        self.inpaint_input_image = inpaint_input_image
        self.inpaint_additional_prompt = inpaint_additional_prompt
        self.image_prompts = image_prompts
        self.save_extension = save_extension
        self.require_base64 = require_base64
        self.advanced_params = advanced_params

        if self.advanced_params is None:
            self.advanced_params = AdvancedParams()

            # Auto set mixing_image_prompt_and_inpaint to True
            if len(self.image_prompts) > 0 and self.inpaint_input_image is not None:
                print("Mixing Image Prompts and Inpaint Enabled")
                self.advanced_params.mixing_image_prompt_and_inpaint = True
            if len(self.image_prompts) > 0 and self.uov_input_image is not None:
                print("Mixing Image Prompts and Vary Upscale Enabled")
                self.advanced_params.mixing_image_prompt_and_vary_upscale = True

    def to_dict(self):
        """
        Convert the ImageGenerationParams object to a dictionary.
        Args:
            self:

        Returns:
            self to dict
        """
        obj_dict = copy.deepcopy(self)
        return obj_dict.__dict__
