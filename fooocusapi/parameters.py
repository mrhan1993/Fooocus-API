from enum import Enum
from typing import Dict, List, Tuple
import numpy as np

from fooocusapi.task_queue import TaskType


inpaint_model_version = 'v1'


defualt_styles = ['Fooocus V2', 'Fooocus Enhance', 'Fooocus Sharp']
default_base_model_name = 'sd_xl_base_1.0_0.9vae.safetensors'
default_refiner_model_name = 'sd_xl_refiner_1.0_0.9vae.safetensors'
default_lora = 'sd_xl_offset_example-lora_1.0.safetensors'
default_lora_weight = 0.5
default_cfg_scale = 7.0
default_prompt_negative = ''


aspect_ratios = [
    '704×1408',
    '704×1344',
    '768×1344',
    '768×1280',
    '832×1216',
    '832×1152',
    '896×1152',
    '896×1088',
    '960×1088',
    '960×1024',
    '1024×1024',
    '1024×960',
    '1088×960',
    '1088×896',
    '1152×896',
    '1152×832',
    '1216×832',
    '1280×768',
    '1344×768',
    '1344×704',
    '1408×704',
    '1472×704',
    '1536×640',
    '1600×640',
    '1664×576',
    '1728×576',
]

uov_methods = [
    'Disabled', 'Vary (Subtle)', 'Vary (Strong)', 'Upscale (1.5x)', 'Upscale (2x)', 'Upscale (Fast 2x)'
]


outpaint_expansions = [
    'Left', 'Right', 'Top', 'Bottom'
]


class GenerationFinishReason(str, Enum):
    success = 'SUCCESS'
    queue_is_full = 'QUEUE_IS_FULL'
    user_cancel = 'USER_CANCEL'
    error = 'ERROR'


class ImageGenerationResult(object):
    def __init__(self, im: str | None, seed: int, finish_reason: GenerationFinishReason):
        self.im = im
        self.seed = seed
        self.finish_reason = finish_reason


class ImageGenerationParams(object):
    def __init__(self, prompt: str,
                 negative_prompt: str,
                 style_selections: List[str],
                 performance_selection: List[str],
                 aspect_ratios_selection: str,
                 image_number: int,
                 image_seed: int | None,
                 sharpness: float,
                 guidance_scale: float,
                 base_model_name: str,
                 refiner_model_name: str,
                 loras: List[Tuple[str, float]],
                 uov_input_image: np.ndarray | None,
                 uov_method: str,
                 outpaint_selections: List[str],
                 inpaint_input_image: Dict[str, np.ndarray] | None,
                 image_prompts: List[Tuple[np.ndarray, float, float, str]]):
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
        self.loras = loras
        self.uov_input_image = uov_input_image
        self.uov_method = uov_method
        self.outpaint_selections = outpaint_selections
        self.inpaint_input_image = inpaint_input_image
        self.image_prompts = image_prompts
