# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from typing import List
from cog import BasePredictor, Input, Path

from fooocusapi.parameters import GenerationFinishReason, ImageGenerationParams
from fooocusapi.worker import process_generate
from PIL import Image


class Args(object):
    sync_repo = None


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        from main import prepare_environments
        prepare_environments(Args())

        import modules.default_pipeline as _
        print("Predictor setuped")

    def predict(
        self,
        prompt: str = Input(
            default='', description="Prompt for image generation")
    ) -> List[Path]:
        """Run a single prediction on the model"""
        from modules.util import generate_temp_filename
        import modules.flags as flags

        negative_promit = ''
        style_selections = ['Fooocus V2', 'Default (Slightly Cinematic)']
        performance_selection = 'Spped'
        aspect_ratios_selection = '1152×896'
        image_number = 1
        image_seed = -1
        sharpness = 2.0
        guidance_scale = 7.0
        base_model_name = 'sd_xl_base_1.0_0.9vae.safetensors'
        refiner_model_name = 'sd_xl_refiner_1.0_0.9vae.safetensors'
        loras = [('sd_xl_offset_example-lora_1.0.safetensors', 0.5)]
        uov_input_image = None
        uov_method = flags.disabled
        outpaint_selections = []
        inpaint_input_image = None
        image_prompts = []

        params = ImageGenerationParams(prompt=prompt,
                                 negative_promit=negative_promit,
                                 style_selections=style_selections,
                                 performance_selection=performance_selection,
                                 aspect_ratios_selection=aspect_ratios_selection,
                                 image_number=image_number,
                                 image_seed=image_seed,
                                 sharpness=sharpness,
                                 guidance_scale=guidance_scale,
                                 base_model_name=base_model_name,
                                 refiner_model_name=refiner_model_name,
                                 loras=loras,
                                 uov_input_image=uov_input_image,
                                 uov_method=uov_method,
                                 outpaint_selections=outpaint_selections,
                                 inpaint_input_image=inpaint_input_image,
                                 image_prompts=image_prompts
                                 )

        text_to_img_req = ImageGenerationParams(prompt=prompt)
        results = process_generate(text_to_img_req)

        output_paths: List[Path] = []
        for r in results:
            if r.finish_reason == GenerationFinishReason.success and r.im is not None:
                output_path = generate_temp_filename('/tmp')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                Image.fromarray(r.im).save(output_path)
                output_paths.append(Path(output_path))

        return output_paths
