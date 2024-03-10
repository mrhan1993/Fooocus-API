import copy
import re

import numpy as np
import torch

import fooocus_version

from extras import (preprocessors, ip_adapter, face_crop)
from extras.expansion import safe_str
from modules import core
from modules.config import downloading_sdxl_lcm_lora
from modules.private_logger import log
from modules.upscaler import perform_upscale
from modules.util import (
    remove_empty_str,
    resize_image,
    set_image_shape_ceil,
    get_image_shape_ceil,
    get_shape_ceil
)
from fooocusapi.models.common.response import GeneratedImageResult
from fooocusapi.workers.utils import (
    check_aspect_ratios,
    refresh_seed,
    steps_based_on_performance,
    process_inpaint_input_image,
    expansion_style,
    overwrite_by_ap,
    determine_task_type,
    process_uov,
    process_inpaint,
    process_image_prompt,
    add_tasks)
from fooocusapi.utils.file_utils import get_file_serve_url, save_output_file
from fooocusapi.utils.img_utils import narray_to_base64img
from fooocusapi.utils.logger import default_logger
from fooocusapi.parameters import ImageGenerationParams


def process_stop():
    """Stop a running task."""
    import ldm_patched.modules.model_management
    ldm_patched.modules.model_management.interrupt_current_processing()


@torch.no_grad()
@torch.inference_mode()
def process_generate(self, params: ImageGenerationParams):
    """
    Process the generation of images.
    :param self: self
    :param params: 任务参数
    :return: 任务结果
    """
    try:
        from modules import default_pipeline as pipeline
    except Exception as e:
        default_logger.std_error(f'[Fooocus API] Import default pipeline error: {e}')
        return {}

    from ldm_patched.modules import model_management
    from modules import (patch, flags, inpaint_worker,
                         advanced_parameters)
    from modules.util import (HWC3, resample_image)

    try:
        # Transform parameters
        prompt = params.prompt
        negative_prompt = params.negative_prompt
        style_selections = params.style_selections
        performance_selection = params.performance_selection
        aspect_ratios_selection = params.aspect_ratios_selection
        image_number = params.image_number
        image_seed = refresh_seed(seed_string=params.image_seed)
        sharpness = params.sharpness
        guidance_scale = params.guidance_scale
        base_model_name = params.base_model_name
        refiner_model_name = params.refiner_model_name
        refiner_switch = params.refiner_switch
        loras = params.loras

        input_image_checkbox = params.uov_input_image is not None or params.inpaint_input_image is not None or len(params.image_prompts) > 0
        current_tab = 'uov' if params.uov_method != flags.disabled else 'ip' if len(params.image_prompts) > 0 else 'inpaint' if params.inpaint_input_image is not None else None
        uov_method = params.uov_method
        upscale_value = params.upscale_value
        uov_input_image = params.uov_input_image
        outpaint_selections = params.outpaint_selections
        outpaint_distance_left = params.outpaint_distance_left
        outpaint_distance_top = params.outpaint_distance_top
        outpaint_distance_right = params.outpaint_distance_right
        outpaint_distance_bottom = params.outpaint_distance_bottom
        inpaint_input_image = params.inpaint_input_image
        inpaint_additional_prompt = '' if params.inpaint_additional_prompt is None else params.inpaint_additional_prompt
        inpaint_mask_image_upload = None

        cn_tasks = {x: [] for x in flags.ip_list}
        for img_prompt in params.image_prompts:
            cn_img, cn_stop, cn_weight, cn_type = img_prompt
            cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        advanced_parameters.set_all_advanced_parameters(*params.advanced_params)

        mask = process_inpaint_input_image(inpaint_input_image)
        if mask:
            inpaint_input_image = mask
            advanced_parameters.inpaint_mask_upload_checkbox = True
            inpaint_mask_image_upload = inpaint_input_image['mask']

        # Fooocus async_worker.py code start
        outpaint_selections = [o.lower() for o in outpaint_selections]
        base_model_additional_loras = []
        raw_style_selections = copy.deepcopy(style_selections)
        uov_method = uov_method.lower()

        use_expansion, use_style = expansion_style(style_selections)

        if base_model_name == refiner_model_name:
            default_logger.std_warn('Base model and refiner model are same. Refiner will be disabled.')
            refiner_model_name = 'None'

        steps = steps_based_on_performance(performance_selection)

        if steps == 8:
            default_logger.std_info('Extreme Speed mode enabled.')
            default_logger.std_info('Refiner disabled in LCM mode.')

            loras += [(downloading_sdxl_lcm_lora(), 1.0)]
            refiner_model_name = 'None'
            advanced_parameters.sampler_name = 'lcm'
            advanced_parameters.scheduler_name = 'lcm'
            patch.sharpness = sharpness = 0.0
            guidance_scale = 1.0
            patch.adaptive_cfg = advanced_parameters.adaptive_cfg = 1.0
            refiner_switch = 1.0
            patch.positive_adm_scale = advanced_parameters.adm_scaler_positive = 1.0
            patch.negative_adm_scale = advanced_parameters.adm_scaler_negative = 1.0
            patch.adm_scaler_end = advanced_parameters.adm_scaler_end = 0.0

        patch.adaptive_cfg = advanced_parameters.adaptive_cfg
        patch.sharpness = sharpness
        patch.positive_adm_scale = advanced_parameters.adm_scaler_positive
        patch.negative_adm_scale = advanced_parameters.adm_scaler_negative
        patch.adm_scaler_end = advanced_parameters.adm_scaler_end
        cfg_scale = float(guidance_scale)

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = check_aspect_ratios(aspect_ratios_selection)

        skip_prompt_processing = False
        refiner_swap_method = advanced_parameters.refiner_swap_method

        inpaint_worker.current_task = None
        inpaint_parameterized = advanced_parameters.inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None
        inpaint_patch_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        seed = int(image_seed)

        default_logger.std_info(f'[Parameters] Adaptive CFG = {patch.adaptive_cfg}')
        default_logger.std_info(f'[Parameters] Sharpness = {patch.sharpness}')
        default_logger.std_info(f'[Parameters] ADM Scale = {patch.positive_adm_scale} : {patch.negative_adm_scale} : {patch.adm_scaler_end}')
        default_logger.std_info(f'[Parameters] CFG = {cfg_scale}')
        default_logger.std_info(f'[Parameters] Seed = {seed}')

        sampler_name = advanced_parameters.sampler_name
        scheduler_name = advanced_parameters.scheduler_name

        goals = []
        tasks = []

        if input_image_checkbox:
            generate_type = determine_task_type(current_tab, uov_method, uov_input_image,
                                                inpaint_input_image, advanced_parameters)

            if generate_type == 'uov':
                uov_input_image = HWC3(uov_input_image)
                goals, skip_prompt_processing, steps = process_uov(uov_method, skip_prompt_processing,
                                                                   performance_selection, steps)

            if generate_type == 'inpaint':
                inpaint_input_image, inpaint_mask, \
                    controlnet_canny_path, controlnet_cpds_path, clip_vision_path, \
                    ip_negative_path, ip_adapter_path, \
                    ip_adapter_face_path = process_inpaint(
                        inpaint_input_image, inpaint_mask_image_upload,
                        outpaint_selections, inpaint_parameterized,
                        inpaint_head_model_path, inpaint_patch_model_path,
                        use_synthetic_refiner, refiner_switch,
                        base_model_additional_loras, refiner_model_name,
                        inpaint_additional_prompt, prompt,
                        advanced_parameters
                    )
            if generate_type == 'ip':
                goals.append('cn')
                controlnet_canny_path, controlnet_cpds_path, clip_vision_path, \
                    ip_negative_path, ip_adapter_path, ip_adapter_face_path = process_image_prompt(cn_tasks)

        # Load or unload CNs
        pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        switch = int(round(steps * refiner_switch))

        steps, switch, width, height = overwrite_by_ap(steps, switch, width,
                                                       height, advanced_parameters)

        default_logger.std_info(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        default_logger.std_info(f'[Parameters] Steps = {steps} - {switch}')

        if not skip_prompt_processing:

            prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
            negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            if prompt == '':
                # disable expansion when empty since it is not meaningful and influences image prompt
                use_expansion = False

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

            pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name,
                                        loras=loras, base_model_additional_loras=base_model_additional_loras,
                                        use_synthetic_refiner=use_synthetic_refiner)

            tasks = add_tasks(image_number, prompt, negative_prompt, seed,
                              extra_positive_prompts, extra_negative_prompts,
                              use_style, style_selections)

            if use_expansion:
                for i, t in enumerate(tasks):
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    default_logger.std_info(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            for i, t in enumerate(tasks):
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if 'vary' in goals:
            if 'subtle' in uov_method:
                denoising_strength = 0.5
            if 'strong' in uov_method:
                denoising_strength = 0.85
            if advanced_parameters.overwrite_vary_strength > 0:
                denoising_strength = advanced_parameters.overwrite_vary_strength

            shape_ceil = get_image_shape_ceil(uov_input_image)
            if shape_ceil < 1024:
                default_logger.std_warn(f'[Vary] Image is resized because it is too small. ({shape_ceil} -> 1024)')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                default_logger.std_warn(f'[Vary] Image is resized because it is too big. ({shape_ceil} -> 2048)')
                shape_ceil = 2048

            uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)

            initial_pixels = core.numpy_to_pytorch(uov_input_image)

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
            b, c, height, width = initial_latent['samples'].shape
            width = width * 8
            height = height * 8
            default_logger.std_info(f'Final resolution is {str((height, width))}.')

        if 'upscale' in goals:
            height, width, channel = uov_input_image.shape
            uov_input_image = perform_upscale(uov_input_image)
            default_logger.std_info(f'Image upscale finished to {str(uov_input_image.shape[0:2])}.')

            f = 1.0
            if upscale_value is not None and upscale_value > 1.0:
                f = upscale_value
            else:
                pattern = r"([0-9]+(?:\.[0-9]+)?)x"
                matches = re.findall(pattern, uov_method)
                if len(matches) > 0:
                    f_tmp = float(matches[0])
                    f = 1.0 if f_tmp < 1.0 else 5.0 if f_tmp > 5.0 else f_tmp

            shape_ceil = get_shape_ceil(height * f, width * f)

            if shape_ceil < 1024:
                default_logger.std_warn(f'[Upscale] Image is resized because it is too small. ({shape_ceil} -> 1024)')
                uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
                shape_ceil = 1024
            else:
                uov_input_image = resample_image(uov_input_image, width=width * f, height=height * f)

            image_is_super_large = shape_ceil > 2800

            if 'fast' in uov_method:
                direct_return = True
            elif image_is_super_large:
                default_logger.std_warn('[Upscale] Image is too large. Directly returned the SR image. '
                                        'Usually directly return SR image at 4K resolution yields better results than SDXL diffusion.')
                direct_return = True
            else:
                direct_return = False

            if direct_return:
                d = [('Upscale (Fast)', '2x')]
                log(uov_input_image, d)
                # return uov_input_image list
                return uov_input_image

            tiled = True
            denoising_strength = 0.382

            if advanced_parameters.overwrite_upscale_strength > 0:
                denoising_strength = advanced_parameters.overwrite_upscale_strength

            initial_pixels = core.numpy_to_pytorch(uov_input_image)

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(
                vae=candidate_vae,
                pixels=initial_pixels, tiled=True)
            b, c, height, width = initial_latent['samples'].shape
            width = width * 8
            height = height * 8
            default_logger.std_info(f'Final resolution is {str((height, width))}.')

        if 'inpaint' in goals:
            if len(outpaint_selections) > 0:
                height, width, channel = inpaint_image.shape
                if 'top' in outpaint_selections:
                    distance_top = int(height * 0.3)
                    if outpaint_distance_top > 0:
                        distance_top = outpaint_distance_top

                    inpaint_image = np.pad(inpaint_image, [[distance_top, 0], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[distance_top, 0], [0, 0]], mode='constant', constant_values=255)

                if 'bottom' in outpaint_selections:
                    distance_bottom = int(height * 0.3)
                    if outpaint_distance_bottom > 0:
                        distance_bottom = outpaint_distance_bottom

                    inpaint_image = np.pad(inpaint_image, [[0, distance_bottom], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, distance_bottom], [0, 0]], mode='constant', constant_values=255)

                height, width, channel = inpaint_image.shape
                if 'left' in outpaint_selections:
                    distance_left = int(width * 0.3)
                    if outpaint_distance_left > 0:
                        distance_left = outpaint_distance_left

                    inpaint_image = np.pad(inpaint_image, [[0, 0], [distance_left, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [distance_left, 0]], mode='constant', constant_values=255)

                if 'right' in outpaint_selections:
                    distance_right = int(width * 0.3)
                    if outpaint_distance_right > 0:
                        distance_right = outpaint_distance_right

                    inpaint_image = np.pad(inpaint_image, [[0, 0], [0, distance_right], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, distance_right]], mode='constant', constant_values=255)

                inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
                advanced_parameters.inpaint_strength = 1.0
                advanced_parameters.inpaint_respective_field = 1.0

            denoising_strength = advanced_parameters.inpaint_strength

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=advanced_parameters.inpaint_respective_field
            )

            if advanced_parameters.debugging_inpaint_preprocessor:
                return inpaint_worker.current_task.visualize_mask_processing()

            inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
            inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
            inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

            candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            latent_inpaint, latent_mask = core.encode_vae_inpaint(
                mask=inpaint_pixel_mask,
                vae=candidate_vae,
                pixels=inpaint_pixel_image)

            latent_swap = None
            if candidate_vae_swap is not None:
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap,
                    pixels=inpaint_pixel_fill)['samples']

            latent_fill = core.encode_vae(
                vae=candidate_vae,
                pixels=inpaint_pixel_fill)['samples']

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

            if inpaint_parameterized:
                pipeline.final_unet = inpaint_worker.current_task.patch(
                    inpaint_head_model_path=inpaint_head_model_path,
                    inpaint_latent=latent_inpaint,
                    inpaint_latent_mask=latent_mask,
                    model=pipeline.final_unet
                )

            if not advanced_parameters.inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}

            b, c, height, width = latent_fill.shape
            height, width = height * 8, width * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            default_logger.std_info(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            for task in cn_tasks[flags.cn_canny]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = preprocessors.canny_pyramid(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    return cn_img
            for task in cn_tasks[flags.cn_cpds]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = preprocessors.cpds(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    return cn_img
            for task in cn_tasks[flags.cn_ip]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
                if advanced_parameters.debugging_cn_preprocessor:
                    return cn_img
            for task in cn_tasks[flags.cn_ip_face]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = face_crop.crop_image(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
                if advanced_parameters.debugging_cn_preprocessor:
                    return cn_img

            all_ip_tasks = cn_tasks[flags.cn_ip] + cn_tasks[flags.cn_ip_face]

            if len(all_ip_tasks) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

        if advanced_parameters.freeu_enabled:
            default_logger.std_info('FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                advanced_parameters.freeu_b1,
                advanced_parameters.freeu_b2,
                advanced_parameters.freeu_s1,
                advanced_parameters.freeu_s2
            )

        all_steps = steps * image_number

        default_logger.std_info(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        default_logger.std_info(f'[Parameters] Initial Latent shape: {log_shape}')

        final_sampler_name = sampler_name
        final_scheduler_name = scheduler_name

        if scheduler_name == 'lcm':
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_refiner_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            default_logger.std_info('Using lcm scheduler.')

        def callback(step, x0, x, total_steps, y):
            # y is preview image, np.int8
            x0, x, total_steps
            p = int((float(step + 1) / float(all_steps)) * 100)
            self.update('status', 'running')
            self.update('progress', p)
            self.update('task_step_preview', narray_to_base64img(y))

        generated_res = []

        for current_task_id, task in enumerate(tasks):
            try:
                positive_cond, negative_cond = task['c'], task['uc']

                if 'cn' in goals:
                    for cn_flag, cn_path in [
                        (flags.cn_canny, controlnet_canny_path),
                        (flags.cn_cpds, controlnet_cpds_path)
                    ]:
                        for cn_img, cn_stop, cn_weight in cn_tasks[cn_flag]:
                            positive_cond, negative_cond = core.apply_controlnet(
                                positive_cond, negative_cond,
                                pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)

                imgs = pipeline.process_diffusion(
                    positive_cond=positive_cond,
                    negative_cond=negative_cond,
                    steps=steps,
                    switch=switch,
                    width=width,
                    height=height,
                    image_seed=task['task_seed'],
                    callback=callback,
                    sampler_name=final_sampler_name,
                    scheduler_name=final_scheduler_name,
                    latent=initial_latent,
                    denoise=denoising_strength,
                    tiled=tiled,
                    cfg_scale=cfg_scale,
                    refiner_swap_method=refiner_swap_method
                )

                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]
                for x in imgs:
                    params = {
                        'Prompt': task['log_positive_prompt'],
                        'NegativePrompt': task['log_negative_prompt'],
                        'FooocusV2Expansion': task['expansion'],
                        'Styles': str(raw_style_selections),
                        'Performance': performance_selection,
                        'Resolution': str((width, height)),
                        'Sharpness': sharpness,
                        'GuidanceScale': guidance_scale,
                        'ADMGuidance': [
                            patch.positive_adm_scale,
                            patch.negative_adm_scale,
                            patch.adm_scaler_end],
                        'BaseModel': base_model_name,
                        'RefinerModel': refiner_model_name,
                        'RefinerSwitch': refiner_switch,
                        'Sampler': sampler_name,
                        'Scheduler': scheduler_name,
                        'Seed': task['task_seed'],
                        'Version': 'v' + fooocus_version.version
                    }
                    list_lora = []
                    for name, weight in loras:
                        if name != 'None':
                            list_lora.append({'name': name, 'weight': weight})
                    params['LoRAs'] = list_lora
                    # log(x, d) # Fooocus 原生保存函数
                    image_name = save_output_file(img=x, image_meta=params,
                                                  image_name=self.task_id + '-' + str(current_task_id))
                    image_url = get_file_serve_url(image_name)
                    b64_img = None
                    if self.req_param.require_base64:
                        b64_img = narray_to_base64img(x)
                    generated_res.append(
                        GeneratedImageResult(
                            base64=b64_img,
                            url=image_url,
                            seed=str(task['task_seed']),
                            finish_reason='SUCCESS'
                        ))
                self.update("task_step_preview", None)
                self.update("task_result", generated_res)
                self.update("status", "completed")
                self.update("task_status", "success")
            except model_management.InterruptProcessingException as e:
                self.update("status", "completed")
                self.update("task_status", "canceled")
                default_logger.std_warn(f"[Fooocus API] User stopped, Info: {e}")
                return []
            except Exception as e:
                self.update("task_status", "failed")
                self.update("status", "error")
                default_logger.std_error(message=f"[Task Queue] Worker error,Info: {e}")
                return []

        return self.to_dict()
    except Exception as e:
        default_logger.std_error(message=f"[Fooocus API] Worker error, {e}")
        self.update("status", "error")
        self.update("task_status", "failed")
        return self.to_dict()
