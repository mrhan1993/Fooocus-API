from enum import Enum
from typing import BinaryIO, Dict, List, Tuple
import numpy as np


class TaskType(str, Enum):
    text2img = 'text2img'


class GenerationFinishReason(str, Enum):
    success = 'SUCCESS'
    queue_is_full = 'QUEUE_IS_FULL'
    user_cancel = 'USER_CANCEL'
    error = 'ERROR'


class ImageGenerationResult(object):
    def __init__(self, im: np.ndarray | None, seed: int, finish_reason: GenerationFinishReason):
        self.im = im
        self.seed = seed
        self.finish_reason = finish_reason


class ImageGenerationParams(object):
    def __init__(self, prompt: str,
                 negative_promit: str,
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
                 uov_input_image: BinaryIO | None,
                 uov_method: str,
                 outpaint_selections: List[str],
                 inpaint_input_image: Dict[str, np.ndarray] | None,
                 image_prompts: List[Tuple[BinaryIO, float, float, str]]):
        self.prompt = prompt
        self.negative_promit = negative_promit
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
