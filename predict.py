"""
Prediction interface for Cog ⚙️
https://github.com/replicate/cog/blob/main/docs/python.md
"""

import copy
import os
from typing import List
import numpy as np

from PIL import Image
from cog import BasePredictor, BaseModel, Input, Path
from fooocusapi.utils.lora_manager import LoraManager
from fooocusapi.utils.file_utils import output_dir
from fooocusapi.models.common.task import GenerationFinishReason
from fooocusapi.configs.default import (
    available_aspect_ratios,
    uov_methods,
    outpaint_expansions,
    default_styles,
    default_base_model_name,
    default_refiner_model_name,
    default_loras,
    default_refiner_switch,
    default_cfg_scale,
    default_prompt_negative
)

from fooocusapi.parameters import ImageGenerationParams
from fooocusapi.task_queue import TaskType


class Output(BaseModel):
    """
    Output model
    """
    seeds: List[str]
    paths: List[Path]


class Predictor(BasePredictor):
    """Predictor"""
    def setup(self) -> None:
        """
        Load the model into memory to make running multiple predictions efficient
        """
        from main import pre_setup
        pre_setup()

    def predict(
        self,
        prompt: str = Input(
            default='',
            description="Prompt for image generation"),
        negative_prompt: str = Input(
            default=default_prompt_negative,
            description="Negative prompt for image generation"),
        style_selections: str = Input(
            default=','.join(default_styles),
            description="Fooocus styles applied for image generation, separated by comma"),
        performance_selection: str = Input(
            default='Speed', 
            choices=['Speed', 'Quality', 'Extreme Speed', 'Lightning'],
            description="Performance selection"),
        aspect_ratios_selection: str = Input(
            default='1152*896',
            choices=available_aspect_ratios,
            description="The generated image's size"),
        image_number: int = Input(
            default=1,
            ge=1, le=8,
            description="How many image to generate"),
        image_seed: int = Input(
            default=-1,
            description="Seed to generate image, -1 for random"),
        use_default_loras: bool = Input(
            default=True,
            description="Use default LoRAs"),
        loras_custom_urls: str = Input(
            default="",
            description="Custom LoRAs URLs in the format 'url,weight' provide multiple separated by ; (example 'url1,0.3;url2,0.1')"),
        sharpness: float = Input(
            default=2.0,
            ge=0.0, le=30.0),
        guidance_scale: float = Input(
            default=default_cfg_scale,
            ge=1.0, le=30.0),
        refiner_switch: float = Input(
            default=default_refiner_switch,
            ge=0.1, le=1.0),
        uov_input_image: Path = Input(
            default=None,
            description="Input image for upscale or variation, keep None for not upscale or variation"),
        uov_method: str = Input(
            default='Disabled',
            choices=uov_methods),
        uov_upscale_value: float = Input(
            default=0,
            description="Only when Upscale (Custom)"),
        inpaint_additional_prompt: str = Input(
            default='',
            description="Prompt for image generation"),
        inpaint_input_image: Path = Input(
            default=None,
            description="Input image for inpaint or outpaint, keep None for not inpaint or outpaint. Please noticed, `uov_input_image` has bigger priority is not None."),
        inpaint_input_mask: Path = Input(
            default=None,
            description="Input mask for inpaint"),
        outpaint_selections: str = Input(
            default='',
            description="Outpaint expansion selections, literal 'Left', 'Right', 'Top', 'Bottom' separated by comma"),
        outpaint_distance_left: int = Input(
            default=0,
            description="Outpaint expansion distance from Left of the image"),
        outpaint_distance_top: int = Input(
            default=0,
            description="Outpaint expansion distance from Top of the image"),
        outpaint_distance_right: int = Input(
            default=0,
            description="Outpaint expansion distance from Right of the image"),
        outpaint_distance_bottom: int = Input(
            default=0,
            description="Outpaint expansion distance from Bottom of the image"),
        cn_img1: Path = Input(
            default=None,
            description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
        cn_stop1: float = Input(
            default=None,
            ge=0, le=1,
            description="Stop at for image prompt, None for default value"),
        cn_weight1: float = Input(
            default=None,
            ge=0, le=2,
            description="Weight for image prompt, None for default value"),
        cn_type1: str = Input(
            default='ImagePrompt',
            choices=['ImagePrompt', 'FaceSwap', 'PyraCanny', 'CPDS'],
            description="ControlNet type for image prompt"),
        cn_img2: Path = Input(
            default=None,
            description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
        cn_stop2: float = Input(
            default=None,
            ge=0, le=1,
            description="Stop at for image prompt, None for default value"),
        cn_weight2: float = Input(
            default=None,
            ge=0, le=2,
            description="Weight for image prompt, None for default value"),
        cn_type2: str = Input(
            default='ImagePrompt',
            choices=['ImagePrompt', 'FaceSwap', 'PyraCanny', 'CPDS'],
            description="ControlNet type for image prompt"),
        cn_img3: Path = Input(
            default=None,
            description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
        cn_stop3: float = Input(
            default=None,
            ge=0, le=1,
            description="Stop at for image prompt, None for default value"),
        cn_weight3: float = Input(
            default=None,
            ge=0, le=2,
            description="Weight for image prompt, None for default value"),
        cn_type3: str = Input(
            default='ImagePrompt',
            choices=['ImagePrompt', 'FaceSwap', 'PyraCanny', 'CPDS'],
            description="ControlNet type for image prompt"),
        cn_img4: Path = Input(
            default=None,
            description="Input image for image prompt. If all cn_img[n] are None, image prompt will not applied."),
        cn_stop4: float = Input(
            default=None,
            ge=0, le=1,
            description="Stop at for image prompt, None for default value"),
        cn_weight4: float = Input(
            default=None,
            ge=0, le=2,
            description="Weight for image prompt, None for default value"),
        cn_type4: str = Input(
            default='ImagePrompt',
            choices=['ImagePrompt', 'FaceSwap', 'PyraCanny', 'CPDS'],
            description="ControlNet type for image prompt")
    ) -> Output:
        """Run a single prediction on the model"""
        from modules import flags
        from modules.sdxl_styles import legal_style_names
        from fooocusapi.worker import blocking_get_task_result, worker_queue

        base_model_name = default_base_model_name
        refiner_model_name = default_refiner_model_name

        lora_manager = LoraManager()
        
        # Use default loras if selected
        loras = copy.copy(default_loras) if use_default_loras else []

        # add custom user loras if provided
        if loras_custom_urls:
            urls = [url.strip() for url in loras_custom_urls.split(';')]

            loras_with_weights = [url.split(',') for url in urls]

            custom_lora_paths = lora_manager.check([lw[0] for lw in loras_with_weights])
            custom_loras = [[path, float(lw[1]) if len(lw) > 1 else 1.0] for path, lw in
                            zip(custom_lora_paths, loras_with_weights)]

            loras.extend(custom_loras)

        style_selections_arr = []
        for s in style_selections.strip().split(','):
            style = s.strip()
            if style in legal_style_names:
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
        image_prompt_config = [
            (cn_img1, cn_stop1, cn_weight1, cn_type1),
            (cn_img2, cn_stop2, cn_weight2, cn_type2),
            (cn_img3, cn_stop3, cn_weight3, cn_type3),
            (cn_img4, cn_stop4, cn_weight4, cn_type4)]
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

        advanced_params = None

        params = ImageGenerationParams(
            prompt=prompt,
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
            refiner_switch=refiner_switch,
            loras=loras,
            uov_input_image=uov_input_image,
            uov_method=uov_method,
            upscale_value=uov_upscale_value,
            outpaint_selections=outpaint_selections_arr,
            inpaint_input_image=inpaint_input_image_dict,
            image_prompts=image_prompts,
            advanced_params=advanced_params,
            inpaint_additional_prompt=inpaint_additional_prompt,
            outpaint_distance_left=outpaint_distance_left,
            outpaint_distance_top=outpaint_distance_top,
            outpaint_distance_right=outpaint_distance_right,
            outpaint_distance_bottom=outpaint_distance_bottom,
            save_meta=True,
            meta_scheme='fooocus',
            save_extension='png',
            save_name='',
            require_base64=False,
        )

        print(f"[Predictor Predict] Params: {params.__dict__}")

        async_task = worker_queue.add_task(
            TaskType.text_2_img,
            params)

        if async_task is None:
            print("[Task Queue] The task queue has reached limit")
            raise Exception("The task queue has reached limit.")

        results = blocking_get_task_result(async_task.job_id)

        output_paths: List[Path] = []
        output_seeds: List[str] = []
        for r in results:
            if r.finish_reason == GenerationFinishReason.success and r.im is not None:
                output_seeds.append(r.seed)
                output_paths.append(Path(os.path.join(output_dir, r.im)))

        print(f"[Predictor Predict] Finished with {len(output_paths)} images")

        if len(output_paths) == 0:
            raise Exception("Process failed.")

        return Output(seeds=output_seeds, paths=output_paths)
