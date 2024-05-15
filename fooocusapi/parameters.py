from typing import Dict, List, Tuple
import numpy as np
import copy

from fooocusapi.models.common.requests import AdvancedParams


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
        save_meta: bool,
        meta_scheme: str,
        save_name: str,
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
        self.save_meta = save_meta
        self.meta_scheme = meta_scheme
        self.save_name = save_name
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
