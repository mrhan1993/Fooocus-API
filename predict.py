# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
from typing import List
from cog import BasePredictor, Input, Path

from fooocusapi.parameters import inpaint_model_version, GenerationFinishReason, ImageGenerationParams, fooocus_styles, aspect_ratios, uov_methods, outpaint_expansions
from fooocusapi.worker import process_generate
import numpy as np
from PIL import Image


class Args(object):
    sync_repo = None


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        from main import pre_setup
        pre_setup(disable_private_log=True, preload_pipeline=True)

    def predict(
        self,
        prompt: str = Input(default='', description="Prompt for image generation"),
        negative_prompt: str = Input(default='', description="Negtive prompt for image generation"),
        style_selections: str = Input(default='Fooocus V2,Default (Slightly Cinematic)', description="Fooocus styles applied for image generation, seperated by comma"),
        performance_selection: str = Input(default='Speed', description="Performance selection", choices=['Speed', 'Quality']),
        aspect_ratios_selection: str = Input(default='1152×896', description="The generated image's size", choices=aspect_ratios),
        image_number: int = Input(default=1, description="How many image to generate", ge=1, le=8),
        image_seed: int = Input(default=-1, description="Seed to generate image, -1 for random"),
        sharpness: float = Input(default=2.0, ge=0.0, le=30.0),
        guidance_scale: float = Input(default=7.0, ge=1.0, le=30.0),
        uov_input_image: Path = Input(default=None, description="Input image for upscale or variation, keep None for not upscale or variation"),
        uov_method: str = Input(default='Disabled', choices=uov_methods),
        inpaint_input_image: Path = Input(default=None, description="Input image for inpaint or outpaint, keep None for not inpaint or outpaint. Please noticed, `uov_input_image` has bigger priority is not None."),
        inpaint_input_mask: Path = Input(default=None, description="Input mask for inpaint"),
        outpaint_selections: str = Input(default='', description="Outpaint expansion selections, literal 'Left', 'Right', 'Top', 'Bottom' seperated by comma"),
        cn_img1: Path = Input(default=None, description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
        cn_stop1: float = Input(default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
        cn_weight1: float = Input(default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
        cn_type1: str = Input(default='Image Prompt', description="ControlNet type for image prompt", choices=['Image Prompt', 'PyraCanny', 'CPDS']),
        cn_img2: Path = Input(default=None, description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
        cn_stop2: float = Input(default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
        cn_weight2: float = Input(default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
        cn_type2: str = Input(default='Image Prompt', description="ControlNet type for image prompt", choices=['Image Prompt', 'PyraCanny', 'CPDS']),
        cn_img3: Path = Input(default=None, description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
        cn_stop3: float = Input(default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
        cn_weight3: float = Input(default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
        cn_type3: str = Input(default='Image Prompt', description="ControlNet type for image prompt", choices=['Image Prompt', 'PyraCanny', 'CPDS']),
        cn_img4: Path = Input(default=None, description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
        cn_stop4: float = Input(default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
        cn_weight4: float = Input(default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
        cn_type4: str = Input(default='Image Prompt', description="ControlNet type for image prompt", choices=['Image Prompt', 'PyraCanny', 'CPDS']),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        from modules.util import generate_temp_filename
        import modules.flags as flags

        base_model_name = 'sd_xl_base_1.0_0.9vae.safetensors'
        refiner_model_name = 'sd_xl_refiner_1.0_0.9vae.safetensors'
        loras = [('sd_xl_offset_example-lora_1.0.safetensors', 0.5)]

        style_selections_arr = []
        for s in style_selections.strip().split(','):
            style = s.strip()
            if style in fooocus_styles:
                style_selections_arr.append(style)

        if uov_input_image is not None:
            im = Image.open(str(uov_input_image))
            uov_input_image = np.array(im)

        inpaint_input_image_dict = None
        if inpaint_input_image is not None:
            im = Image.open(str(inpaint_input_image))
            inpaint_input_image = np.array(im)

            if inpaint_input_mask is not None:
                im = Image.open(str(inpaint_input_mask))
                inpaint_input_mask = np.array(im)
            else:
                inpaint_input_mask = np.zeros(inpaint_input_image.shape)
            
            inpaint_input_image_dict = {
                'image': inpaint_input_image,
                'mask': inpaint_input_mask
            }

        outpaint_selections_arr = []
        for e in outpaint_selections.strip().split(','):
            expansion = e.strip()
            if expansion in outpaint_expansions:
                outpaint_selections_arr.append(expansion)

        image_prompts = []
        image_prompt_config = [(cn_img1, cn_stop1, cn_weight1, cn_type1), (cn_img2, cn_stop2, cn_weight2, cn_type2),
                               (cn_img3, cn_stop3, cn_weight3, cn_type3), (cn_img4, cn_stop4, cn_weight4, cn_type4)]
        for config in image_prompt_config:
            cn_img, cn_stop, cn_weight, cn_type = config
            if cn_img is not None:
                im = Image.open(str(cn_img))
                cn_img = np.array(im)
                if cn_stop is None:
                    cn_stop = flags.default_parameters[cn_type][0]
                if cn_weight is None:
                    cn_weight = flags.default_parameters[cn_type][1]
                image_prompts.append((cn_img, cn_stop, cn_weight, cn_type))


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
                                 outpaint_selections=outpaint_selections_arr,
                                 inpaint_input_image=inpaint_input_image_dict,
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
