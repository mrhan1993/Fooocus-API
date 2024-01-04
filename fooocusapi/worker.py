import copy
import random
import time
import numpy as np
import torch
import re
import logging

from typing import List
from fooocusapi.file_utils import save_output_file
from fooocusapi.parameters import GenerationFinishReason, ImageGenerationParams, ImageGenerationResult
from fooocusapi.task_queue import QueueTask, TaskQueue, TaskOutputs


task_queue = TaskQueue(queue_size=3, hisotry_size=6, webhook_url=None)


def process_top():
    import ldm_patched.modules.model_management
    ldm_patched.modules.model_management.interrupt_current_processing()


@torch.no_grad()
@torch.inference_mode()
def process_generate(async_task: QueueTask, params: ImageGenerationParams) -> List[ImageGenerationResult]:
    try:
        import modules.default_pipeline as pipeline
    except Exception as e:
        print('Import default pipeline error:', e)
        if not async_task.is_finished:
            task_queue.finish_task(async_task.job_id)
            async_task.set_result([], True, str(e))
            print(f"[Task Queue] Finish task with error, seq={async_task.job_id}")
        return []

    import modules.patch as patch
    import modules.flags as flags
    import modules.core as core
    import modules.inpaint_worker as inpaint_worker
    import modules.config as config
    import modules.advanced_parameters as advanced_parameters
    import modules.constants as constants
    import extras.preprocessors as preprocessors
    import extras.ip_adapter as ip_adapter
    import extras.face_crop as face_crop
    import ldm_patched.modules.model_management as model_management
    from modules.util import remove_empty_str, resize_image, HWC3, set_image_shape_ceil, get_image_shape_ceil, get_shape_ceil, resample_image, erode_or_dilate
    from modules.private_logger import log
    from modules.upscaler import perform_upscale
    from extras.expansion import safe_str
    from modules.sdxl_styles import apply_style, fooocus_expansion, apply_wildcards
    import fooocus_version

    outputs = TaskOutputs(async_task)
    results = []

    def refresh_seed(r, seed_string):
        if r:
            return random.randint(constants.MIN_SEED, constants.MAX_SEED)
        else:
            try:
                seed_value = int(seed_string)
                if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                    return seed_value
            except ValueError:
                pass
            return random.randint(constants.MIN_SEED, constants.MAX_SEED)
        
    def progressbar(_, number, text):
        print(f'[Fooocus] {text}')
        outputs.append(['preview', (number, text, None)])

    def yield_result(_, imgs, tasks):
        if not isinstance(imgs, list):
            imgs = [imgs]

        results = []
        for i, im in enumerate(imgs):
            seed = -1 if len(tasks) == 0 else tasks[i]['task_seed']
            img_filename = save_output_file(im)
            results.append(ImageGenerationResult(im=img_filename, seed=str(seed), finish_reason=GenerationFinishReason.success))
        async_task.set_result(results, False)
        task_queue.finish_task(async_task.job_id)
        print(f"[Task Queue] Finish task, job_id={async_task.job_id}")

        outputs.append(['results', imgs])
        pipeline.prepare_text_encoder(async_call=True)
        return results

    try:
        waiting_sleep_steps: int = 0
        waiting_start_time = time.perf_counter()
        while not task_queue.is_task_ready_to_start(async_task.job_id):
            if waiting_sleep_steps == 0:
                print(
                    f"[Task Queue] Waiting for task queue become free, job_id={async_task.job_id}")
            delay = 0.1
            time.sleep(delay)
            waiting_sleep_steps += 1
            if waiting_sleep_steps % int(10 / delay) == 0:
                waiting_time = time.perf_counter() - waiting_start_time
                print(
                    f"[Task Queue] Already waiting for {waiting_time}S, seq={async_task.job_id}")

        print(f"[Task Queue] Task queue is free, start task, job_id={async_task.job_id}")

        task_queue.start_task(async_task.job_id)

        execution_start_time = time.perf_counter()

        # Transform pamameters
        prompt = params.prompt
        negative_prompt = params.negative_prompt
        style_selections = params.style_selections
        performance_selection = params.performance_selection
        aspect_ratios_selection = params.aspect_ratios_selection
        image_number = params.image_number
        image_seed = None if params.image_seed == -1 else params.image_seed
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
        inpaint_additional_prompt = params.inpaint_additional_prompt
        inpaint_mask_image_upload = None

        if inpaint_additional_prompt is None:
            inpaint_additional_prompt = ''

        image_seed = refresh_seed(image_seed is None, image_seed)

        cn_tasks = {x: [] for x in flags.ip_list}
        for img_prompt in params.image_prompts:
            cn_img, cn_stop, cn_weight, cn_type = img_prompt
            cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        advanced_parameters.set_all_advanced_parameters(*params.advanced_params)

        if inpaint_input_image is not None and inpaint_input_image['image'] is not None:
            inpaint_image_size = inpaint_input_image['image'].shape[:2]
            if inpaint_input_image['mask'] is None:
                inpaint_input_image['mask'] = np.zeros(inpaint_image_size, dtype=np.uint8)
            else:
                advanced_parameters.inpaint_mask_upload_checkbox = True

            inpaint_input_image['mask'] = HWC3(inpaint_input_image['mask'])
            inpaint_mask_image_upload = inpaint_input_image['mask']

        # Fooocus async_worker.py code start

        outpaint_selections = [o.lower() for o in outpaint_selections]
        base_model_additional_loras = []
        raw_style_selections = copy.deepcopy(style_selections)
        uov_method = uov_method.lower()

        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0

        if base_model_name == refiner_model_name:
            print(f'Refiner disabled because base model and refiner are same.')
            refiner_model_name = 'None'

        assert performance_selection in ['Speed', 'Quality', 'Extreme Speed']

        steps = 30

        if performance_selection == 'Speed':
            steps = 30

        if performance_selection == 'Quality':
            steps = 60

        if performance_selection == 'Extreme Speed':
            print('Enter LCM mode.')
            progressbar(async_task, 1, 'Downloading LCM components ...')
            loras += [(config.downloading_sdxl_lcm_lora(), 1.0)]

            if refiner_model_name != 'None':
                print(f'Refiner disabled in LCM mode.')

            refiner_model_name = 'None'
            sampler_name = advanced_parameters.sampler_name = 'lcm'
            scheduler_name = advanced_parameters.scheduler_name = 'lcm'
            patch.sharpness = sharpness = 0.0
            cfg_scale = guidance_scale = 1.0
            patch.adaptive_cfg = advanced_parameters.adaptive_cfg = 1.0
            refiner_switch = 1.0
            patch.positive_adm_scale = advanced_parameters.adm_scaler_positive = 1.0
            patch.negative_adm_scale = advanced_parameters.adm_scaler_negative = 1.0
            patch.adm_scaler_end = advanced_parameters.adm_scaler_end = 0.0
            steps = 8

        patch.adaptive_cfg = advanced_parameters.adaptive_cfg
        print(f'[Parameters] Adaptive CFG = {patch.adaptive_cfg}')

        patch.sharpness = sharpness
        print(f'[Parameters] Sharpness = {patch.sharpness}')

        patch.positive_adm_scale = advanced_parameters.adm_scaler_positive
        patch.negative_adm_scale = advanced_parameters.adm_scaler_negative
        patch.adm_scaler_end = advanced_parameters.adm_scaler_end
        print(f'[Parameters] ADM Scale = '
              f'{patch.positive_adm_scale} : '
              f'{patch.negative_adm_scale} : '
              f'{patch.adm_scaler_end}')

        cfg_scale = float(guidance_scale)
        print(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        # Validate input format
        if not aspect_ratios_selection.replace('*', ' ').replace(' ', '').isdigit():
            raise ValueError("Invalid input format. Please enter aspect ratios in the form 'width*height'.")
        width, height = aspect_ratios_selection.replace('*', '*').replace('*', ' ').split(' ')[:2]
        # Validate width and height are integers
        if not (width.isdigit() and height.isdigit()):
            raise ValueError("Invalid width or height. Please enter valid integers.")

        width, height = int(width), int(height)

        skip_prompt_processing = False
        refiner_swap_method = advanced_parameters.refiner_swap_method

        inpaint_worker.current_task = None
        inpaint_parameterized = advanced_parameters.inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        seed = int(image_seed)
        print(f'[Parameters] Seed = {seed}')

        sampler_name = advanced_parameters.sampler_name
        scheduler_name = advanced_parameters.scheduler_name

        goals = []
        tasks = []

        if input_image_checkbox:
            if (current_tab == 'uov' or (
                    current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_vary_upscale)) \
                    and uov_method != flags.disabled and uov_input_image is not None:
                uov_input_image = HWC3(uov_input_image)
                if 'vary' in uov_method:
                    goals.append('vary')
                elif 'upscale' in uov_method:
                    goals.append('upscale')
                    if 'fast' in uov_method:
                        skip_prompt_processing = True
                    else:
                        steps = 18

                        if performance_selection == 'Speed':
                            steps = 18

                        if performance_selection == 'Quality':
                            steps = 36

                        if performance_selection == 'Extreme Speed':
                            steps = 8

                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    config.downloading_upscale_model()
            if (current_tab == 'inpaint' or (
                    current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_inpaint)) \
                    and isinstance(inpaint_input_image, dict):
                inpaint_image = inpaint_input_image['image']
                inpaint_mask = inpaint_input_image['mask'][:, :, 0]

                if advanced_parameters.inpaint_mask_upload_checkbox:
                    if isinstance(inpaint_mask_image_upload, np.ndarray):
                        if inpaint_mask_image_upload.ndim == 3:
                            H, W, C = inpaint_image.shape
                            inpaint_mask_image_upload = resample_image(inpaint_mask_image_upload, width=W, height=H)
                            inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
                            inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
                            inpaint_mask = inpaint_mask_image_upload

                if int(advanced_parameters.inpaint_erode_or_dilate) != 0:
                    inpaint_mask = erode_or_dilate(inpaint_mask, advanced_parameters.inpaint_erode_or_dilate)

                if advanced_parameters.invert_mask_checkbox:
                    inpaint_mask = 255 - inpaint_mask

                inpaint_image = HWC3(inpaint_image)
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    config.downloading_upscale_model()
                    if inpaint_parameterized:
                        progressbar(async_task, 1, 'Downloading inpainter ...')
                        inpaint_head_model_path, inpaint_patch_model_path = config.downloading_inpaint_models(
                            advanced_parameters.inpaint_engine)
                        base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                        print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                        if refiner_model_name == 'None':
                            use_synthetic_refiner = True
                            refiner_switch = 0.5
                    else:
                        inpaint_head_model_path, inpaint_patch_model_path = None, None
                        print(f'[Inpaint] Parameterized inpaint is disabled.')
                    if inpaint_additional_prompt != '':
                        if prompt == '':
                            prompt = inpaint_additional_prompt
                        else:
                            prompt = inpaint_additional_prompt + '\n' + prompt
                    goals.append('inpaint')
            if current_tab == 'ip' or \
                    advanced_parameters.mixing_image_prompt_and_inpaint or \
                    advanced_parameters.mixing_image_prompt_and_vary_upscale:
                goals.append('cn')
                progressbar(async_task, 1, 'Downloading control models ...')
                if len(cn_tasks[flags.cn_canny]) > 0:
                    controlnet_canny_path = config.downloading_controlnet_canny()
                if len(cn_tasks[flags.cn_cpds]) > 0:
                    controlnet_cpds_path = config.downloading_controlnet_cpds()
                if len(cn_tasks[flags.cn_ip]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_path = config.downloading_ip_adapters('ip')
                if len(cn_tasks[flags.cn_ip_face]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_face_path = config.downloading_ip_adapters(
                        'face')
                progressbar(async_task, 1, 'Loading control models ...')

        # Load or unload CNs
        pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        switch = int(round(steps * refiner_switch))

        if advanced_parameters.overwrite_step > 0:
            steps = advanced_parameters.overwrite_step

        if advanced_parameters.overwrite_switch > 0:
            switch = advanced_parameters.overwrite_switch

        if advanced_parameters.overwrite_width > 0:
            width = advanced_parameters.overwrite_width

        if advanced_parameters.overwrite_height > 0:
            height = advanced_parameters.overwrite_height

        print(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        print(f'[Parameters] Steps = {steps} - {switch}')

        progressbar(async_task, 1, 'Initializing ...')

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

            progressbar(async_task, 3, 'Loading models ...')
            pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name,
                                        loras=loras, base_model_additional_loras=base_model_additional_loras,
                                        use_synthetic_refiner=use_synthetic_refiner)

            progressbar(async_task, 3, 'Processing prompts ...')
            tasks = []
            for i in range(image_number):
                task_seed = (seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not
                task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future

                task_prompt = apply_wildcards(prompt, task_rng)
                task_negative_prompt = apply_wildcards(negative_prompt, task_rng)
                task_extra_positive_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_positive_prompts]
                task_extra_negative_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_negative_prompts]

                positive_basic_workloads = []
                negative_basic_workloads = []

                if use_style:
                    for s in style_selections:
                        p, n = apply_style(s, positive=task_prompt)
                        positive_basic_workloads = positive_basic_workloads + p
                        negative_basic_workloads = negative_basic_workloads + n
                else:
                    positive_basic_workloads.append(task_prompt)

                negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

                positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
                negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

                positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
                negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

                tasks.append(dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    expansion='',
                    c=None,
                    uc=None,
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                    log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
                ))

            if use_expansion:
                for i, t in enumerate(tasks):
                    progressbar(async_task, 5, f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    print(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            for i, t in enumerate(tasks):
                progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if len(goals) > 0:
            progressbar(async_task, 13, 'Image processing ...')

        if 'vary' in goals:
            if 'subtle' in uov_method:
                denoising_strength = 0.5
            if 'strong' in uov_method:
                denoising_strength = 0.85
            if advanced_parameters.overwrite_vary_strength > 0:
                denoising_strength = advanced_parameters.overwrite_vary_strength

            shape_ceil = get_image_shape_ceil(uov_input_image)
            if shape_ceil < 1024:
                print(f'[Vary] Image is resized because it is too small.')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                print(f'[Vary] Image is resized because it is too big.')
                shape_ceil = 2048

            uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'upscale' in goals:
            H, W, C = uov_input_image.shape
            progressbar(async_task, 13, f'Upscaling image from {str((H, W))} ...')
            uov_input_image = perform_upscale(uov_input_image)
            print(f'Image upscaled.')

            f = 1.0
            if upscale_value is not None and upscale_value > 1.0:
                f = upscale_value
            else:
                pattern = r"([0-9]+(?:\.[0-9]+)?)x"
                matches = re.findall(pattern, uov_method)
                if len(matches) > 0:
                    f_tmp = float(matches[0])
                    f = 1.0 if f_tmp < 1.0 else 5.0 if f_tmp > 5.0 else f_tmp

            shape_ceil = get_shape_ceil(H * f, W * f)

            if shape_ceil < 1024:
                print(f'[Upscale] Image is resized because it is too small.')
                uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
                shape_ceil = 1024
            else:
                uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)

            image_is_super_large = shape_ceil > 2800

            if 'fast' in uov_method:
                direct_return = True
            elif image_is_super_large:
                print('Image is too large. Directly returned the SR image. '
                      'Usually directly return SR image at 4K resolution '
                      'yields better results than SDXL diffusion.')
                direct_return = True
            else:
                direct_return = False

            if direct_return:
                d = [('Upscale (Fast)', '2x')]
                log(uov_input_image, d)
                return yield_result(async_task, uov_input_image, tasks)

            tiled = True
            denoising_strength = 0.382

            if advanced_parameters.overwrite_upscale_strength > 0:
                denoising_strength = advanced_parameters.overwrite_upscale_strength

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(
                vae=candidate_vae,
                pixels=initial_pixels, tiled=True)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'inpaint' in goals:
            if len(outpaint_selections) > 0:
                H, W, C = inpaint_image.shape
                if 'top' in outpaint_selections:
                    distance_top = int(H * 0.3)
                    if outpaint_distance_top > 0:
                        distance_top = outpaint_distance_top

                    inpaint_image = np.pad(inpaint_image, [[distance_top, 0], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[distance_top, 0], [0, 0]], mode='constant',
                                          constant_values=255)
                
                if 'bottom' in outpaint_selections:
                    distance_bottom = int(H * 0.3)
                    if outpaint_distance_bottom > 0:
                        distance_bottom = outpaint_distance_bottom

                    inpaint_image = np.pad(inpaint_image, [[0, distance_bottom], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, distance_bottom], [0, 0]], mode='constant',
                                          constant_values=255)

                H, W, C = inpaint_image.shape
                if 'left' in outpaint_selections:
                    distance_left = int(W * 0.3)
                    if outpaint_distance_left > 0:
                        distance_left = outpaint_distance_left
                    
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [distance_left, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [distance_left, 0]], mode='constant',
                                          constant_values=255)
                
                if 'right' in outpaint_selections:
                    distance_right = int(W * 0.3)
                    if outpaint_distance_right > 0:
                        distance_right = outpaint_distance_right
                    
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [0, distance_right], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, distance_right]], mode='constant',
                                          constant_values=255)
                    
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
                return yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(),
                             do_not_show_finished_images=True)

            progressbar(async_task, 13, 'VAE Inpaint encoding ...')

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
                progressbar(async_task, 13, 'VAE SD15 encoding ...')
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap,
                    pixels=inpaint_pixel_fill)['samples']

            progressbar(async_task, 13, 'VAE encoding ...')
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

            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            for task in cn_tasks[flags.cn_canny]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = preprocessors.canny_pyramid(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    return yield_result(async_task, cn_img, tasks)
            for task in cn_tasks[flags.cn_cpds]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = preprocessors.cpds(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    return yield_result(async_task, cn_img, tasks)
            for task in cn_tasks[flags.cn_ip]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
                if advanced_parameters.debugging_cn_preprocessor:
                    return yield_result(async_task, cn_img, tasks)
            for task in cn_tasks[flags.cn_ip_face]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = face_crop.crop_image(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
                if advanced_parameters.debugging_cn_preprocessor:
                    return yield_result(async_task, cn_img, tasks)

            all_ip_tasks = cn_tasks[flags.cn_ip] + cn_tasks[flags.cn_ip_face]

            if len(all_ip_tasks) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

        if advanced_parameters.freeu_enabled:
            print(f'FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                advanced_parameters.freeu_b1,
                advanced_parameters.freeu_b2,
                advanced_parameters.freeu_s1,
                advanced_parameters.freeu_s2
            )

        all_steps = steps * image_number

        print(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        print(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

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
            print('Using lcm scheduler.')

        outputs.append(['preview', (13, 'Moving model to GPU ...', None)])

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * steps + step
            outputs.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}-th Sampling',
                y)])

        for current_task_id, task in enumerate(tasks):
            execution_start_time = time.perf_counter()

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
                    d = [
                        ('Prompt', task['log_positive_prompt']),
                        ('Negative Prompt', task['log_negative_prompt']),
                        ('Fooocus V2 Expansion', task['expansion']),
                        ('Styles', str(raw_style_selections)),
                        ('Performance', performance_selection),
                        ('Resolution', str((width, height))),
                        ('Sharpness', sharpness),
                        ('Guidance Scale', guidance_scale),
                        ('ADM Guidance', str((
                            patch.positive_adm_scale,
                            patch.negative_adm_scale,
                            patch.adm_scaler_end))),
                        ('Base Model', base_model_name),
                        ('Refiner Model', refiner_model_name),
                        ('Refiner Switch', refiner_switch),
                        ('Sampler', sampler_name),
                        ('Scheduler', scheduler_name),
                        ('Seed', task['task_seed']),
                    ]
                    for n, w in loras:
                        if n != 'None':
                            d.append((f'LoRA', f'{n} : {w}'))
                    d.append(('Version', 'v' + fooocus_version.version))
                    log(x, d)
                
                # Fooocus async_worker.py code end
                
                results += imgs
            except model_management.InterruptProcessingException as e:
                print("User stopped")
                results.append(ImageGenerationResult(
                    im=None, seed=task['task_seed'], finish_reason=GenerationFinishReason.user_cancel))
                async_task.set_result(results, True, str(e))
                break
            except Exception as e:
                print('Process error:', e)
                results.append(ImageGenerationResult(
                    im=None, seed=task['task_seed'], finish_reason=GenerationFinishReason.error))
                async_task.set_result(results, True, str(e))
                break

            execution_time = time.perf_counter() - execution_start_time
            print(f'Generating and saving time: {execution_time:.2f} seconds')

        if async_task.finish_with_error:
            task_queue.finish_task(async_task.job_id)
            return async_task.task_result
        return yield_result(None, results, tasks)
    except Exception as e:
        print('Worker error:', e)
        logging.exception(e)

        if not async_task.is_finished:
            task_queue.finish_task(async_task.job_id)
            async_task.set_result([], True, str(e))
            print(f"[Task Queue] Finish task with error, job_id={async_task.job_id}")
        return []
