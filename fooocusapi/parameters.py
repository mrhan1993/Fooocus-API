from typing import Dict, List, Tuple
import numpy as np
import copy

from fooocusapi.models.common.base import EnhanceCtrlNets
from fooocusapi.models.common.requests import AdvancedParams
from modules import config


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
        loras: List[Tuple[bool, str, float]],
        uov_input_image: np.ndarray | None,
        uov_method: str,
        upscale_value: float | None,
        outpaint_selections: List[str],
        outpaint_distance_left: int,
        outpaint_distance_right: int,
        outpaint_distance_top: int,
        outpaint_distance_bottom: int,
        inpaint_input_image: Dict[str, np.ndarray | None],
        inpaint_additional_prompt: str | None,
        enhance_input_image: np.ndarray | None,
        enhance_checkbox: bool,
        enhance_uov_method: str,
        enhance_uov_processing_order,
        enhance_uov_prompt_type,
        save_final_enhanced_image_only,
        enhance_ctrlnets: List[EnhanceCtrlNets],
        image_prompts: List[Tuple[np.ndarray, float, float, str]],
        read_wildcards_in_order: bool,
        advanced_params: AdvancedParams | None,
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
        self.loras = loras[:config.default_max_lora_number] if len(loras) > config.default_max_lora_number else loras
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
        self.image_prompts = image_prompts[:config.default_controlnet_image_count] if len(image_prompts) > config.default_controlnet_image_count else image_prompts
        self.enhance_input_image = enhance_input_image
        self.enhance_checkbox = enhance_checkbox
        self.enhance_uov_method = enhance_uov_method
        self.enhance_uov_processing_order = enhance_uov_processing_order
        self.enhance_uov_prompt_type = enhance_uov_prompt_type
        self.save_final_enhanced_image_only = save_final_enhanced_image_only
        self.enhance_ctrlnets = enhance_ctrlnets[:config.default_enhance_tabs] if len(enhance_ctrlnets) > config.default_enhance_tabs else enhance_ctrlnets
        self.current_tab = None
        self.read_wildcards_in_order = read_wildcards_in_order
        self.save_extension = save_extension
        self.save_meta = save_meta
        self.meta_scheme = meta_scheme
        self.save_name = save_name
        self.require_base64 = require_base64
        self.advanced_params = advanced_params

        self.current_tab = 'uov'
        if self.enhance_input_image is not None:
            self.current_tab = 'enhance'
        elif self.image_prompts[0][0] is not None:
            self.current_tab = 'ip'
        elif self.uov_input_image is not None:
            self.current_tab = 'uov'
        elif self.inpaint_input_image["image"] is not None:
            self.current_tab = 'inpaint'

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
