# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
from typing import List
from cog import BasePredictor, Input, Path

from fooocusapi.parameters import GenerationFinishReason, ImageGenerationParams, fooocus_styles, aspect_ratios
from fooocusapi.worker import process_generate
from PIL import Image


class Args(object):
    sync_repo = None


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        from main import prepare_environments
        print("[Predictor Setup] Prepare environments")
        prepare_environments(Args())

        print("[Predictor Setup] Preload pipeline")
        import modules.default_pipeline as _
        print("[Predictor Setup] Finished")

    def predict(
        self,
        prompt: str = Input(default='', description="Prompt for image generation"),
        negative_prompt: str = Input(default='', description="Negtive prompt for image generation"),
        style_selections: str = Input(default='Fooocus V2,Default (Slightly Cinematic)', description="Fooocus styles applied for image generation, seperated by comma"),
        performance_selection: str = Input(default='Speed', description="Performance selection", choices=['Speed', 'Quality']),
        aspect_ratios_selection: str = Input(default='1152x896', description="The generated image's size", choices=aspect_ratios),
        image_number: int = Input(default=1, description="How many image to generate", ge=1, le=8),
        image_seed: int = Input(default=-1, description="Seed to generate image, -1 for random"),
        sharpness: float = Input(default=2.0, ge=0.0, le=30.0),
        guidance_scale: float = Input(default=7.0, ge=1.0, le=30.0),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        from modules.util import generate_temp_filename
        import modules.flags as flags

        aspect_ratios_selection = aspect_ratios_selection.replace('x', '×')

        base_model_name = 'sd_xl_base_1.0_0.9vae.safetensors'
        refiner_model_name = 'sd_xl_refiner_1.0_0.9vae.safetensors'
        loras = [('sd_xl_offset_example-lora_1.0.safetensors', 0.5)]
        uov_input_image = None
        uov_method = flags.disabled
        outpaint_selections = []
        inpaint_input_image = None
        image_prompts = []

        style_selections_arr = []
        for s in style_selections.strip().split(','):
            if s in fooocus_styles:
                style = s.strip()
                if style in fooocus_styles:
                    style_selections_arr.append(style)

        params = ImageGenerationParams(prompt=prompt,
                                 negative_prompt=negative_prompt,
                                 style_selections=style_selections_arr,
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
        
        print(f"[Predictor Predict] Params: {params.__dict__}")

        results = process_generate(params)

        output_paths: List[Path] = []
        for r in results:
            if r.finish_reason == GenerationFinishReason.success and r.im is not None:
                _, local_temp_filename, _ = generate_temp_filename('/tmp')
                os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)
                Image.fromarray(r.im).save(local_temp_filename)
                output_paths.append(Path(local_temp_filename))

        print(f"[Predictor Predict] Finished with {len(output_paths)} images")

        if len(output_paths) == 0:
            raise Exception(
                f"Process failed."
            )

        return output_paths
