"""
requests models for v1 endpoints
"""
from typing import List
from fastapi.params import File
from fastapi import (
    UploadFile,
    Form
)
from fooocusapi.models.common.requests import (
    CommonRequest,
    advanced_params_parser
)
from fooocusapi.models.common.base import (
    EnhanceCtrlNets, ImagePrompt,
    ControlNetType,
    MaskModel, OutpaintExpansion,
    UpscaleOrVaryMethod,
    PerformanceSelection
)

from fooocusapi.models.common.base import (
    style_selection_parser,
    lora_parser,
    outpaint_selections_parser,
    image_prompt_parser,
    default_loras_json
)

from fooocusapi.configs.default import (
    default_prompt_negative,
    default_aspect_ratio,
    default_base_model_name,
    default_refiner_model_name,
    default_refiner_switch,
    default_cfg_scale,
    default_styles,
)


class ImgUpscaleOrVaryRequest(CommonRequest):
    """
    Request for image upscale or variation
    Attributes:
        input_image: Input image
        uov_method: Upscale or variation method
        upscale_value: upscale value
    Functions:
        as_form: Convert request to form data
    """
    input_image: UploadFile
    uov_method: UpscaleOrVaryMethod
    upscale_value: float | None

    @classmethod
    def as_form(
            cls,
            input_image: UploadFile = Form(description="Init image for upscale or outpaint"),
            uov_method: UpscaleOrVaryMethod = Form(),
            upscale_value: float | None = Form(None, description="Upscale custom value, None for default value", ge=1.0, le=5.0),
            prompt: str = Form(''),
            negative_prompt: str = Form(default_prompt_negative),
            style_selections: List[str] = Form(default_styles, description="Fooocus style selections, separated by comma"),
            performance_selection: PerformanceSelection = Form(PerformanceSelection.speed, description="Performance Selection, one of 'Speed','Quality','Extreme Speed'"),
            aspect_ratios_selection: str = Form(default_aspect_ratio, description="Aspect Ratios Selection, default 1152*896"),
            image_number: int = Form(default=1, description="Image number", ge=1, le=32),
            image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
            sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
            guidance_scale: float = Form(default=default_cfg_scale, ge=1.0, le=30.0),
            base_model_name: str = Form(default_base_model_name, description="checkpoint file name"),
            refiner_model_name: str = Form(default_refiner_model_name, description="refiner file name"),
            refiner_switch: float = Form(default=default_refiner_switch, description="Refiner Switch At", ge=0.1, le=1.0),
            loras: str | None = Form(default=default_loras_json, description='Lora config in JSON. Format as [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}]'),
            advanced_params: str | None = Form(default=None, description="Advanced parameters in JSON"),
            save_meta: bool = Form(default=False, description="Save metadata to image"),
            meta_scheme: str = Form(default='fooocus', description="Metadata scheme, one of 'fooocus', 'a111'"),
            save_extension: str = Form(default="png", description="Save extension, png, jpg or webp"),
            save_name: str = Form(default="", description="Save name, empty for auto generate"),
            require_base64: bool = Form(default=False, description="Return base64 data of generated image"),
            read_wildcards_in_order: bool = Form(default=False, description="Read wildcards in order"),
            async_process: bool = Form(default=False, description="Set to true will run async and return job info for retrieve generation result later"),
            webhook_url: str = Form(default="", description="Webhook url for generation result"),
    ):
        style_selection_arr = style_selection_parser(style_selections)
        loras_model = lora_parser(loras)
        advanced_params_obj = advanced_params_parser(advanced_params)

        return cls(
            input_image=input_image, uov_method=uov_method, upscale_value=upscale_value,
            prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
            performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
            image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
            base_model_name=base_model_name, refiner_model_name=refiner_model_name, refiner_switch=refiner_switch,
            loras=loras_model, advanced_params=advanced_params_obj, save_meta=save_meta, meta_scheme=meta_scheme,
            save_extension=save_extension, save_name=save_name, require_base64=require_base64,
            read_wildcards_in_order=read_wildcards_in_order, async_process=async_process, webhook_url=webhook_url)


class ImgInpaintOrOutpaintRequest(CommonRequest):
    """
    Image Inpaint or Outpaint Request
    """
    input_image: UploadFile | None
    input_mask: UploadFile | None
    inpaint_additional_prompt: str | None
    outpaint_selections: List[OutpaintExpansion]
    outpaint_distance_left: int
    outpaint_distance_right: int
    outpaint_distance_top: int
    outpaint_distance_bottom: int

    @classmethod
    def as_form(
            cls,
            input_image: UploadFile = Form(description="Init image for inpaint or outpaint"),
            input_mask: UploadFile = Form(File(None), description="Inpaint or outpaint mask"),
            inpaint_additional_prompt: str | None = Form("", description="Describe what you want to inpaint"),
            outpaint_selections: List[str] = Form([], description="Outpaint expansion selections, literal 'Left', 'Right', 'Top', 'Bottom' separated by comma"),
            outpaint_distance_left: int = Form(default=0, description="Set outpaint left distance, -1 for default"),
            outpaint_distance_right: int = Form(default=0, description="Set outpaint right distance, -1 for default"),
            outpaint_distance_top: int = Form(default=0, description="Set outpaint top distance, -1 for default"),
            outpaint_distance_bottom: int = Form(default=0, description="Set outpaint bottom distance, -1 for default"),
            prompt: str = Form(''),
            negative_prompt: str = Form(default_prompt_negative),
            style_selections: List[str] = Form(default_styles, description="Fooocus style selections, separated by comma"),
            performance_selection: PerformanceSelection = Form(PerformanceSelection.speed, description="Performance Selection, one of 'Speed','Quality','Extreme Speed'"),
            aspect_ratios_selection: str = Form(default_aspect_ratio, description="Aspect Ratios Selection, default 1152*896"),
            image_number: int = Form(default=1, description="Image number", ge=1, le=32),
            image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
            sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
            guidance_scale: float = Form(default=default_cfg_scale, ge=1.0, le=30.0),
            base_model_name: str = Form(default_base_model_name),
            refiner_model_name: str = Form(default_refiner_model_name),
            refiner_switch: float = Form(default=default_refiner_switch, description="Refiner Switch At", ge=0.1, le=1.0),
            loras: str | None = Form(default=default_loras_json, description='Lora config in JSON. Format as [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}]'),
            advanced_params: str | None = Form(default=None, description="Advanced parameters in JSON"),
            save_meta: bool = Form(default=False, description="Save metadata to image"),
            meta_scheme: str = Form(default='fooocus', description="Metadata scheme, one of 'fooocus', 'a111'"),
            save_extension: str = Form(default="png", description="Save extension, png, jpg or webp"),
            save_name: str = Form(default="", description="Save name, empty for auto generate"),
            require_base64: bool = Form(default=False, description="Return base64 data of generated image"),
            read_wildcards_in_order: bool = Form(default=False, description="Read wildcards in order"),
            async_process: bool = Form(default=False, description="Set to true will run async and return job info for retrieve generation result later"),
            webhook_url: str = Form(default="", description="Webhook url for generation result"),
    ):
        if isinstance(input_mask, File):
            input_mask = None

        outpaint_selections_arr = outpaint_selections_parser(outpaint_selections)
        style_selection_arr = style_selection_parser(style_selections)
        loras_model = lora_parser(loras)
        advanced_params_obj = advanced_params_parser(advanced_params)

        return cls(
            input_image=input_image, input_mask=input_mask, inpaint_additional_prompt=inpaint_additional_prompt,
            outpaint_selections=outpaint_selections_arr, outpaint_distance_left=outpaint_distance_left,
            outpaint_distance_right=outpaint_distance_right, outpaint_distance_top=outpaint_distance_top,
            outpaint_distance_bottom=outpaint_distance_bottom, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
            performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
            image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
            base_model_name=base_model_name, refiner_model_name=refiner_model_name, refiner_switch=refiner_switch,
            loras=loras_model, advanced_params=advanced_params_obj, save_meta=save_meta, meta_scheme=meta_scheme,
            save_extension=save_extension, save_name=save_name, require_base64=require_base64,
            read_wildcards_in_order=read_wildcards_in_order, async_process=async_process, webhook_url=webhook_url)


class ImgPromptRequest(ImgInpaintOrOutpaintRequest):
    """
    Image Prompt Request
    """
    image_prompts: List[ImagePrompt]

    @classmethod
    def as_form(
            cls,
            input_image: UploadFile = Form(File(None), description="Init image for inpaint or outpaint"),
            input_mask: UploadFile = Form(File(None), description="Inpaint or outpaint mask"),
            inpaint_additional_prompt: str | None = Form(default='', description="Describe what you want to inpaint"),
            outpaint_selections: List[str] = Form([], description="Outpaint expansion selections, literal 'Left', 'Right', 'Top', 'Bottom' separated by comma"),
            outpaint_distance_left: int = Form(default=0, description="Set outpaint left distance, 0 for default"),
            outpaint_distance_right: int = Form(default=0, description="Set outpaint right distance, 0 for default"),
            outpaint_distance_top: int = Form(default=0, description="Set outpaint top distance, 0 for default"),
            outpaint_distance_bottom: int = Form(default=0, description="Set outpaint bottom distance, 0 for default"),
            cn_img1: UploadFile = Form(File(None), description="Input image for image prompt"),
            cn_stop1: float | None = Form(
                default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
            cn_weight1: float | None = Form(
                default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
            cn_type1: ControlNetType = Form(
                default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
            cn_img2: UploadFile = Form(
                File(None), description="Input image for image prompt"),
            cn_stop2: float | None = Form(
                default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
            cn_weight2: float | None = Form(
                default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
            cn_type2: ControlNetType = Form(
                default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
            cn_img3: UploadFile = Form(
                File(None), description="Input image for image prompt"),
            cn_stop3: float | None = Form(
                default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
            cn_weight3: float | None = Form(
                default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
            cn_type3: ControlNetType = Form(
                default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
            cn_img4: UploadFile = Form(
                File(None), description="Input image for image prompt"),
            cn_stop4: float | None = Form(
                default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
            cn_weight4: float | None = Form(
                default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
            cn_type4: ControlNetType = Form(
                default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),

            prompt: str = Form(''),
            negative_prompt: str = Form(default_prompt_negative),
            style_selections: List[str] = Form(default_styles, description="Fooocus style selections, separated by comma"),
            performance_selection: PerformanceSelection = Form(PerformanceSelection.speed),
            aspect_ratios_selection: str = Form(default_aspect_ratio),
            image_number: int = Form(default=1, description="Image number", ge=1, le=32),
            image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
            sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
            guidance_scale: float = Form(default=default_cfg_scale, ge=1.0, le=30.0),
            base_model_name: str = Form(default_base_model_name),
            refiner_model_name: str = Form(default_refiner_model_name),
            refiner_switch: float = Form(default=default_refiner_switch, description="Refiner Switch At", ge=0.1, le=1.0),
            loras: str | None = Form(default=default_loras_json, description='Lora config in JSON. Format as [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}]'),
            advanced_params: str | None = Form(default=None, description="Advanced parameters in JSON"),
            save_meta: bool = Form(default=False, description="Save metadata to image"),
            meta_scheme: str = Form(default='fooocus', description="Metadata scheme, one of 'fooocus', 'a111'"),
            save_extension: str = Form(default="png", description="Save extension, png, jpg or webp"),
            save_name: str = Form(default="", description="Save name, empty for auto generate"),
            require_base64: bool = Form(default=False, description="Return base64 data of generated image"),
            read_wildcards_in_order: bool = Form(default=False, description="Read wildcards in order"),
            async_process: bool = Form(default=False, description="Set to true will run async and return job info for retrieve generation result later"),
            webhook_url: str = Form(default="", description="Webhook url for generation result"),
    ):
        if isinstance(input_image, File):
            input_image = None
        if isinstance(input_mask, File):
            input_mask = None
        if isinstance(cn_img1, File):
            cn_img1 = None
        if isinstance(cn_img2, File):
            cn_img2 = None
        if isinstance(cn_img3, File):
            cn_img3 = None
        if isinstance(cn_img4, File):
            cn_img4 = None

        outpaint_selections_arr = outpaint_selections_parser(outpaint_selections)

        image_prompt_config = [
            (cn_img1, cn_stop1, cn_weight1, cn_type1),
            (cn_img2, cn_stop2, cn_weight2, cn_type2),
            (cn_img3, cn_stop3, cn_weight3, cn_type3),
            (cn_img4, cn_stop4, cn_weight4, cn_type4)]
        image_prompts = image_prompt_parser(image_prompt_config)
        style_selection_arr = style_selection_parser(style_selections)
        loras_model = lora_parser(loras)
        advanced_params_obj = advanced_params_parser(advanced_params)

        return cls(
            input_image=input_image, input_mask=input_mask, inpaint_additional_prompt=inpaint_additional_prompt, outpaint_selections=outpaint_selections_arr,
            outpaint_distance_left=outpaint_distance_left, outpaint_distance_right=outpaint_distance_right, outpaint_distance_top=outpaint_distance_top, outpaint_distance_bottom=outpaint_distance_bottom,
            image_prompts=image_prompts, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
            performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
            image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
            base_model_name=base_model_name, refiner_model_name=refiner_model_name, refiner_switch=refiner_switch,
            loras=loras_model, advanced_params=advanced_params_obj, save_meta=save_meta, meta_scheme=meta_scheme,
            save_extension=save_extension, save_name=save_name, require_base64=require_base64,
            read_wildcards_in_order=read_wildcards_in_order, async_process=async_process, webhook_url=webhook_url)


class ImageEnhanceRequest(CommonRequest):
    """
    Image Enhance Request
    """
    enhance_input_image: UploadFile = Form(File(None), description="Input image for enhance")
    enhance_checkbox: bool = Form(default=True, description="Enhance checkbox")
    enhance_uov_method: UpscaleOrVaryMethod = Form(default=UpscaleOrVaryMethod.strong_variation, description="Upscale or vary method")
    enhance_uov_processing_order: str = Form(default="Before First Enhancement", description="Upscale or vary processing order")
    enhance_uov_prompt_type: str = Form(default="Original Prompts", description="Upscale or vary prompt type")
    save_final_enhanced_image_only: bool = Form(True, description="Save Final Enhanced Image Only")
    enhance_ctrlnets: List[EnhanceCtrlNets]

    @classmethod
    def as_form(
            cls,
            enhance_input_image: UploadFile = Form(File(None), description="Input image for enhance"),
            enhance_checkbox: bool = Form(default=True, description="Enhance checkbox"),
            enhance_uov_method: UpscaleOrVaryMethod = Form(default=UpscaleOrVaryMethod.strong_variation, description="Upscale or vary method"),
            enhance_uov_processing_order: str = Form(default="Before First Enhancement", description="Upscale or vary processing order"),
            enhance_uov_prompt_type: str = Form(default="Original Prompts", description="Upscale or vary prompt type"),

            enhance_enabled_1: bool = Form(default=False, description="Enhance checkbox 1"),
            enhance_mask_dino_prompt_1: str = Form(default="", description="Mask dino prompt"),
            enhance_prompt_1: str = Form(default="", description="Prompt"),
            enhance_negative_prompt_1: str = Form(default="", description="Negative prompt"),
            enhance_mask_model_1: MaskModel = Form(default=MaskModel.sam, description="Mask model"),
            enhance_mask_cloth_category_1: str = Form(default="full", description="Mask cloth category"),
            enhance_mask_sam_model_1: str = Form(default="vit_b", description="one of vit_b vit_h vit_l"),
            enhance_mask_text_threshold_1: float = Form(default=0.25, ge=0, le=1, description="Mask text threshold"),
            enhance_mask_box_threshold_1: float = Form(default=0.3, ge=0, le=1, description="Mask box threshold"),
            enhance_mask_sam_max_detections_1: int = Form(default=0, ge=0, le=10, description="Mask sam max detections, Set to 0 to detect all"),
            enhance_inpaint_disable_initial_latent_1: bool = Form(default=False, description="Inpaint disable initial latent"),
            enhance_inpaint_engine_1: str = Form(default="v2.6", description="Inpaint engine"),
            enhance_inpaint_strength_1: float = Form(default=1, ge=0, le=1, description="Inpaint strength"),
            enhance_inpaint_respective_field_1: float = Form(default=0.618, ge=0, le=1, description="Inpaint respective field"),
            enhance_inpaint_erode_or_dilate_1: float = Form(default=0, ge=-64, le=64, description="Inpaint erode or dilate"),
            enhance_mask_invert_1: bool = Form(default=False, description="Inpaint mask invert"),

            enhance_enabled_2: bool = Form(default=False, description="Enhance checkbox 2"),
            enhance_mask_dino_prompt_2: str = Form(default="", description="Mask dino prompt"),
            enhance_prompt_2: str = Form(default="", description="Prompt"),
            enhance_negative_prompt_2: str = Form(default="", description="Negative prompt"),
            enhance_mask_model_2: MaskModel = Form(default=MaskModel.sam, description="Mask model"),
            enhance_mask_cloth_category_2: str = Form(default="full", description="Mask cloth category"),
            enhance_mask_sam_model_2: str = Form(default="vit_b", description="one of vit_b vit_h vit_l"),
            enhance_mask_text_threshold_2: float = Form(default=0.25, ge=0, le=1, description="Mask text threshold"),
            enhance_mask_box_threshold_2: float = Form(default=0.3, ge=0, le=1, description="Mask box threshold"),
            enhance_mask_sam_max_detections_2: int = Form(default=0, ge=0, le=10, description="Mask sam max detections, Set to 0 to detect all"),
            enhance_inpaint_disable_initial_latent_2: bool = Form(default=False, description="Inpaint disable initial latent"),
            enhance_inpaint_engine_2: str = Form(default="v2.6", description="Inpaint engine"),
            enhance_inpaint_strength_2: float = Form(default=1, ge=0, le=1, description="Inpaint strength"),
            enhance_inpaint_respective_field_2: float = Form(default=0.618, ge=0, le=1, description="Inpaint respective field"),
            enhance_inpaint_erode_or_dilate_2: float = Form(default=0, ge=-64, le=64, description="Inpaint erode or dilate"),
            enhance_mask_invert_2: bool = Form(default=False, description="Inpaint mask invert"),

            enhance_enabled_3: bool = Form(default=False, description="Enhance checkbox 3"),
            enhance_mask_dino_prompt_3: str = Form(default="", description="Mask dino prompt"),
            enhance_prompt_3: str = Form(default="", description="Prompt"),
            enhance_negative_prompt_3: str = Form(default="", description="Negative prompt"),
            enhance_mask_model_3: MaskModel = Form(default=MaskModel.sam, description="Mask model"),
            enhance_mask_cloth_category_3: str = Form(default="full", description="Mask cloth category"),
            enhance_mask_sam_model_3: str = Form(default="vit_b", description="one of vit_b vit_h vit_l"),
            enhance_mask_text_threshold_3: float = Form(default=0.25, ge=0, le=1, description="Mask text threshold"),
            enhance_mask_box_threshold_3: float = Form(default=0.3, ge=0, le=1, description="Mask box threshold"),
            enhance_mask_sam_max_detections_3: int = Form(default=0, ge=0, le=10, description="Mask sam max detections, Set to 0 to detect all"),
            enhance_inpaint_disable_initial_latent_3: bool = Form(default=False, description="Inpaint disable initial latent"),
            enhance_inpaint_engine_3: str = Form(default="v2.6", description="Inpaint engine"),
            enhance_inpaint_strength_3: float = Form(default=1, ge=0, le=1, description="Inpaint strength"),
            enhance_inpaint_respective_field_3: float = Form(default=0.618, ge=0, le=1, description="Inpaint respective field"),
            enhance_inpaint_erode_or_dilate_3: float = Form(default=0, ge=-64, le=64, description="Inpaint erode or dilate"),
            enhance_mask_invert_3: bool = Form(default=False, description="Inpaint mask invert"),

            prompt: str = Form(''),
            negative_prompt: str = Form(default_prompt_negative),
            style_selections: List[str] = Form(default_styles, description="Fooocus style selections, separated by comma"),
            performance_selection: PerformanceSelection = Form(PerformanceSelection.speed),
            aspect_ratios_selection: str = Form(default_aspect_ratio),
            image_number: int = Form(default=1, description="Image number", ge=1, le=32),
            image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
            sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
            guidance_scale: float = Form(default=default_cfg_scale, ge=1.0, le=30.0),
            base_model_name: str = Form(default_base_model_name),
            refiner_model_name: str = Form(default_refiner_model_name),
            refiner_switch: float = Form(default=default_refiner_switch, description="Refiner Switch At", ge=0.1, le=1.0),
            loras: str | None = Form(default=default_loras_json, description='Lora config in JSON. Format as [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}]'),
            advanced_params: str | None = Form(default=None, description="Advanced parameters in JSON"),
            save_meta: bool = Form(default=False, description="Save metadata to image"),
            meta_scheme: str = Form(default='fooocus', description="Metadata scheme, one of 'fooocus', 'a111'"),
            save_extension: str = Form(default="png", description="Save extension, png, jpg or webp"),
            save_name: str = Form(default="", description="Save name, empty for auto generate"),
            require_base64: bool = Form(default=False, description="Return base64 data of generated image"),
            read_wildcards_in_order: bool = Form(default=False, description="Read wildcards in order"),
            async_process: bool = Form(default=False, description="Set to true will run async and return job info for retrieve generation result later"),
            webhook_url: str = Form(default="", description="Webhook url for generation result")
    ):
        style_selection_arr = style_selection_parser(style_selections)
        loras_model = lora_parser(loras)
        advanced_params_obj = advanced_params_parser(advanced_params)
        enhance_ctrlnets = [
            EnhanceCtrlNets(
                enhance_enabled=enhance_enabled_1,
                enhance_mask_dino_prompt=enhance_mask_dino_prompt_1,
                enhance_prompt=enhance_prompt_1,
                enhance_negative_prompt=enhance_negative_prompt_1,
                enhance_mask_model=enhance_mask_model_1,
                enhance_mask_cloth_category=enhance_mask_cloth_category_1,
                enhance_mask_sam_model=enhance_mask_sam_model_1,
                enhance_mask_text_threshold=enhance_mask_text_threshold_1,
                enhance_mask_box_threshold=enhance_mask_box_threshold_1,
                enhance_mask_sam_max_detections=enhance_mask_sam_max_detections_1,
                enhance_inpaint_disable_initial_latent=enhance_inpaint_disable_initial_latent_1,
                enhance_inpaint_engine=enhance_inpaint_engine_1,
                enhance_inpaint_strength=enhance_inpaint_strength_1,
                enhance_inpaint_respective_field=enhance_inpaint_respective_field_1,
                enhance_inpaint_erode_or_dilate=enhance_inpaint_erode_or_dilate_1,
                enhance_mask_invert=enhance_mask_invert_1),
            EnhanceCtrlNets(
                enhance_enabled=enhance_enabled_2,
                enhance_mask_dino_prompt=enhance_mask_dino_prompt_2,
                enhance_prompt=enhance_prompt_2,
                enhance_negative_prompt=enhance_negative_prompt_2,
                enhance_mask_model=enhance_mask_model_2,
                enhance_mask_cloth_category=enhance_mask_cloth_category_2,
                enhance_mask_sam_model=enhance_mask_sam_model_2,
                enhance_mask_text_threshold=enhance_mask_text_threshold_2,
                enhance_mask_box_threshold=enhance_mask_box_threshold_2,
                enhance_mask_sam_max_detections=enhance_mask_sam_max_detections_2,
                enhance_inpaint_disable_initial_latent=enhance_inpaint_disable_initial_latent_2,
                enhance_inpaint_engine=enhance_inpaint_engine_2,
                enhance_inpaint_strength=enhance_inpaint_strength_2,
                enhance_inpaint_respective_field=enhance_inpaint_respective_field_2,
                enhance_inpaint_erode_or_dilate=enhance_inpaint_erode_or_dilate_2,
                enhance_mask_invert=enhance_mask_invert_2),
            EnhanceCtrlNets(
                enhance_enabled=enhance_enabled_3,
                enhance_mask_dino_prompt=enhance_mask_dino_prompt_3,
                enhance_prompt=enhance_prompt_3,
                enhance_negative_prompt=enhance_negative_prompt_3,
                enhance_mask_model=enhance_mask_model_3,
                enhance_mask_cloth_category=enhance_mask_cloth_category_3,
                enhance_mask_sam_model=enhance_mask_sam_model_3,
                enhance_mask_text_threshold=enhance_mask_text_threshold_3,
                enhance_mask_box_threshold=enhance_mask_box_threshold_3,
                enhance_mask_sam_max_detections=enhance_mask_sam_max_detections_3,
                enhance_inpaint_disable_initial_latent=enhance_inpaint_disable_initial_latent_3,
                enhance_inpaint_engine=enhance_inpaint_engine_3,
                enhance_inpaint_strength=enhance_inpaint_strength_3,
                enhance_inpaint_respective_field=enhance_inpaint_respective_field_3,
                enhance_inpaint_erode_or_dilate=enhance_inpaint_erode_or_dilate_3,
                enhance_mask_invert=enhance_mask_invert_3
            ),
        ]
        return cls(
            enhance_input_image=enhance_input_image,
            enhance_checkbox=enhance_checkbox,
            enhance_uov_method=enhance_uov_method,
            enhance_uov_processing_order=enhance_uov_processing_order,
            enhance_uov_prompt_type=enhance_uov_prompt_type,
            enhance_ctrlnets=enhance_ctrlnets,
            prompt=prompt, negative_prompt=negative_prompt,
            style_selections=style_selection_arr,
            performance_selection=performance_selection,
            aspect_ratios_selection=aspect_ratios_selection,
            image_number=image_number,
            image_seed=image_seed,
            sharpness=sharpness,
            guidance_scale=guidance_scale,
            base_model_name=base_model_name,
            refiner_model_name=refiner_model_name,
            refiner_switch=refiner_switch,
            loras=loras_model, advanced_params=advanced_params_obj,
            save_meta=save_meta, meta_scheme=meta_scheme,
            save_extension=save_extension, save_name=save_name,
            require_base64=require_base64,
            read_wildcards_in_order=read_wildcards_in_order,
            async_process=async_process, webhook_url=webhook_url)
