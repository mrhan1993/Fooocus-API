"""
Worker, modify from https://github.com/lllyasviel/Fooocus/blob/main/modules/async_worker.py
"""
import copy
import os
import random
import time
from typing import List
import logging
import numpy as np
import torch

from fooocusapi.models.common.image_meta import image_parse
from modules.patch import PatchSettings, patch_settings, patch_all
from modules.flags import Performance

from fooocusapi.utils.file_utils import save_output_file
from fooocusapi.models.common.task import (
    GenerationFinishReason,
    ImageGenerationResult
)
from fooocusapi.utils.logger import logger
from fooocusapi.task_queue import (
    QueueTask,
    TaskQueue,
    TaskOutputs
)

patch_all()

worker_queue: TaskQueue | None = None
last_model_name = None


def process_stop():
    """Stop process"""
    import ldm_patched.modules.model_management
    ldm_patched.modules.model_management.interrupt_current_processing()


@torch.no_grad()
@torch.inference_mode()
def task_schedule_loop():
    """Task schedule loop"""
    while True:
        if len(worker_queue.queue) == 0:
            time.sleep(0.05)
            continue

        current_task = worker_queue.queue[0]
        if current_task.start_mills == 0:
            process_generate(current_task)


@torch.no_grad()
@torch.inference_mode()
def blocking_get_task_result(job_id: str) -> List[ImageGenerationResult]:
    """
    Get task result, when async_task is false
    :param job_id:
    :return:
    """
    waiting_sleep_steps: int = 0
    waiting_start_time = time.perf_counter()
    while not worker_queue.is_task_finished(job_id):
        if waiting_sleep_steps == 0:
            logger.std_info(f"[Task Queue] Waiting for task finished, job_id={job_id}")
        delay = 0.05
        time.sleep(delay)
        waiting_sleep_steps += 1
        if waiting_sleep_steps % int(10 / delay) == 0:
            waiting_time = time.perf_counter() - waiting_start_time
            logger.std_info(f"[Task Queue] Already waiting for {round(waiting_time, 1)} seconds, job_id={job_id}")

    task = worker_queue.get_task(job_id, True)
    return task.task_result


@torch.no_grad()
@torch.inference_mode()
def process_generate(async_task: QueueTask):
    """Generate image"""
    try:
        import modules.default_pipeline as pipeline
    except Exception as e:
        logger.std_error(f'[Task Queue] Import default pipeline error: {e}')
        if not async_task.is_finished:
            worker_queue.finish_task(async_task.job_id)
            async_task.set_result([], True, str(e))
            logger.std_error(f"[Task Queue] Finish task with error, seq={async_task.job_id}")
        return []

    import modules.flags as flags
    import modules.core as core
    import modules.inpaint_worker as inpaint_worker
    import modules.config as config
    import modules.constants as constants
    import extras.preprocessors as preprocessors
    import extras.ip_adapter as ip_adapter
    import extras.face_crop as face_crop
    import ldm_patched.modules.model_management as model_management
    from modules.util import (
        remove_empty_str, HWC3, resize_image,
        get_image_shape_ceil, set_image_shape_ceil,
        get_shape_ceil, resample_image, erode_or_dilate,
        get_enabled_loras, parse_lora_references_from_prompt, apply_wildcards,
        remove_performance_lora
    )

    from modules.upscaler import perform_upscale
    from extras.expansion import safe_str
    from extras.censor import default_censor
    from modules.sdxl_styles import (
        apply_style, get_random_style,
        fooocus_expansion, apply_arrays, random_style_name
    )

    pid = os.getpid()

    outputs = TaskOutputs(async_task)
    results = []

    def refresh_seed(seed_string: int | str | None) -> int:
        """
        Refresh and check seed number.
        :params seed_string: seed, str or int. None means random
        :return: seed number
        """
        if seed_string is None or seed_string == -1:
            return random.randint(constants.MIN_SEED, constants.MAX_SEED)

        try:
            seed_value = int(seed_string)
            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                return seed_value
        except ValueError:
            pass
        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

    def progressbar(_, number, text):
        """progress bar"""
        logger.std_info(f'[Fooocus] {text}')
        outputs.append(['preview', (number, text, None)])

    def yield_result(_, images, tasks, extension='png',
                     blockout_nsfw=False, censor=True):
        """
        Yield result
        :param _: async task object
        :param images: list for generated image
        :param tasks: the image was generated one by one, when image number is not one, it will be a task list
        :param extension: extension for saved image
        :param blockout_nsfw: blockout nsfw image
        :param censor: censor image
        :return:
        """
        if not isinstance(images, list):
            images = [images]

        if censor and (config.default_black_out_nsfw or black_out_nsfw):
            images = default_censor(images)

        results = []
        for index, im in enumerate(images):
            if async_task.req_param.save_name == '':
                image_name = f"{async_task.job_id}-{str(index)}"
            else:
                image_name = f"{async_task.req_param.save_name}-{str(index)}"
            if len(tasks) == 0:
                img_seed = -1
                img_meta = {}
            else:
                img_seed = tasks[index]['task_seed']
                img_meta = image_parse(
                    async_tak=async_task,
                    task=tasks[index])
            img_filename = save_output_file(
                img=im,
                image_name=image_name,
                image_meta=img_meta,
                extension=extension)
            results.append(ImageGenerationResult(
                im=img_filename,
                seed=str(img_seed),
                finish_reason=GenerationFinishReason.success))
        async_task.set_result(results, False)
        worker_queue.finish_task(async_task.job_id)
        logger.std_info(f"[Task Queue] Finish task, job_id={async_task.job_id}")

        outputs.append(['results', images])
        pipeline.prepare_text_encoder(async_call=True)

    try:
        logger.std_info(f"[Task Queue] Task queue start task, job_id={async_task.job_id}")
        # clear memory
        global last_model_name

        if last_model_name is None:
            last_model_name = async_task.req_param.base_model_name
        if last_model_name != async_task.req_param.base_model_name:
            model_management.cleanup_models()  # key1
            model_management.unload_all_models()
            model_management.soft_empty_cache()  # key2
            last_model_name = async_task.req_param.base_model_name

        worker_queue.start_task(async_task.job_id)

        execution_start_time = time.perf_counter()

        # Transform parameters
        params = async_task.req_param
        prompt = params.prompt
        negative_prompt = params.negative_prompt
        style_selections = params.style_selections
        performance_selection = Performance(params.performance_selection)
        aspect_ratios_selection = params.aspect_ratios_selection
        image_number = params.image_number
        save_metadata_to_images = params.save_meta
        metadata_scheme = params.meta_scheme
        save_extension = params.save_extension
        save_name = params.save_name
        image_seed = refresh_seed(params.image_seed)
        read_wildcards_in_order = False
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

        adp = params.advanced_params
        disable_preview = adp.disable_preview
        disable_intermediate_results = adp.disable_intermediate_results
        disable_seed_increment = adp.disable_seed_increment
        adm_scaler_positive = adp.adm_scaler_positive
        adm_scaler_negative = adp.adm_scaler_negative
        adm_scaler_end = adp.adm_scaler_end
        adaptive_cfg = adp.adaptive_cfg
        sampler_name = adp.sampler_name
        scheduler_name = adp.scheduler_name
        overwrite_step = adp.overwrite_step
        overwrite_switch = adp.overwrite_switch
        overwrite_width = adp.overwrite_width
        overwrite_height = adp.overwrite_height
        overwrite_vary_strength = adp.overwrite_vary_strength
        overwrite_upscale_strength = adp.overwrite_upscale_strength
        mixing_image_prompt_and_vary_upscale = adp.mixing_image_prompt_and_vary_upscale
        mixing_image_prompt_and_inpaint = adp.mixing_image_prompt_and_inpaint
        debugging_cn_preprocessor = adp.debugging_cn_preprocessor
        skipping_cn_preprocessor = adp.skipping_cn_preprocessor
        canny_low_threshold = adp.canny_low_threshold
        canny_high_threshold = adp.canny_high_threshold
        refiner_swap_method = adp.refiner_swap_method
        controlnet_softness = adp.controlnet_softness
        freeu_enabled = adp.freeu_enabled
        freeu_b1 = adp.freeu_b1
        freeu_b2 = adp.freeu_b2
        freeu_s1 = adp.freeu_s1
        freeu_s2 = adp.freeu_s2
        debugging_inpaint_preprocessor = adp.debugging_inpaint_preprocessor
        inpaint_disable_initial_latent = adp.inpaint_disable_initial_latent
        inpaint_engine = adp.inpaint_engine
        inpaint_strength = adp.inpaint_strength
        inpaint_respective_field = adp.inpaint_respective_field
        inpaint_mask_upload_checkbox = adp.inpaint_mask_upload_checkbox
        invert_mask_checkbox = adp.invert_mask_checkbox
        inpaint_erode_or_dilate = adp.inpaint_erode_or_dilate
        black_out_nsfw = adp.black_out_nsfw
        vae_name = adp.vae_name
        clip_skip = adp.clip_skip

        cn_tasks = {x: [] for x in flags.ip_list}
        for img_prompt in params.image_prompts:
            cn_img, cn_stop, cn_weight, cn_type = img_prompt
            cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        if inpaint_input_image is not None and inpaint_input_image['image'] is not None:
            inpaint_image_size = inpaint_input_image['image'].shape[:2]
            if inpaint_input_image['mask'] is None:
                inpaint_input_image['mask'] = np.zeros(inpaint_image_size, dtype=np.uint8)
            else:
                inpaint_mask_upload_checkbox = True

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
            logger.std_warn('[Fooocus] Refiner disabled because base model and refiner are same.')
            refiner_model_name = 'None'

        steps = performance_selection.steps()

        performance_loras = []

        if performance_selection == Performance.EXTREME_SPEED:
            logger.std_warn('[Fooocus] Enter LCM mode.')
            progressbar(async_task, 1, 'Downloading LCM components ...')
            performance_loras += [(config.downloading_sdxl_lcm_lora(), 1.0)]

            if refiner_model_name != 'None':
                logger.std_info('[Fooocus] Refiner disabled in LCM mode.')

            refiner_model_name = 'None'
            sampler_name = 'lcm'
            scheduler_name = 'lcm'
            sharpness = 0.0
            guidance_scale = 1.0
            adaptive_cfg = 1.0
            refiner_switch = 1.0
            adm_scaler_positive = 1.0
            adm_scaler_negative = 1.0
            adm_scaler_end = 0.0

        elif performance_selection == Performance.LIGHTNING:
            logger.std_info('[Fooocus] Enter Lightning mode.')
            progressbar(async_task, 1, 'Downloading Lightning components ...')
            performance_loras += [(config.downloading_sdxl_lightning_lora(), 1.0)]

            if refiner_model_name != 'None':
                logger.std_info('[Fooocus] Refiner disabled in Lightning mode.')

            refiner_model_name = 'None'
            sampler_name = 'euler'
            scheduler_name = 'sgm_uniform'
            sharpness = 0.0
            guidance_scale = 1.0
            adaptive_cfg = 1.0
            refiner_switch = 1.0
            adm_scaler_positive = 1.0
            adm_scaler_negative = 1.0
            adm_scaler_end = 0.0

        elif performance_selection == Performance.HYPER_SD:
            print('Enter Hyper-SD mode.')
            progressbar(async_task, 1, 'Downloading Hyper-SD components ...')
            performance_loras += [(config.downloading_sdxl_hyper_sd_lora(), 0.8)]

            if refiner_model_name != 'None':
                logger.std_info('[Fooocus] Refiner disabled in Hyper-SD mode.')

            refiner_model_name = 'None'
            sampler_name = 'dpmpp_sde_gpu'
            scheduler_name = 'karras'
            sharpness = 0.0
            guidance_scale = 1.0
            adaptive_cfg = 1.0
            refiner_switch = 1.0
            adm_scaler_positive = 1.0
            adm_scaler_negative = 1.0
            adm_scaler_end = 0.0

        logger.std_info(f'[Parameters] Adaptive CFG = {adaptive_cfg}')
        logger.std_info(f'[Parameters] CLIP Skip = {clip_skip}')
        logger.std_info(f'[Parameters] Sharpness = {sharpness}')
        logger.std_info(f'[Parameters] ControlNet Softness = {controlnet_softness}')
        logger.std_info(f'[Parameters] ADM Scale = '
                        f'{adm_scaler_positive} : '
                        f'{adm_scaler_negative} : '
                        f'{adm_scaler_end}')

        patch_settings[pid] = PatchSettings(
            sharpness,
            adm_scaler_end,
            adm_scaler_positive,
            adm_scaler_negative,
            controlnet_softness,
            adaptive_cfg
        )

        cfg_scale = float(guidance_scale)
        logger.std_info(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = aspect_ratios_selection.replace('×', ' ').replace('*', ' ').split(' ')[:2]
        width, height = int(width), int(height)

        skip_prompt_processing = False

        inpaint_worker.current_task = None
        inpaint_parameterized = inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        seed = int(image_seed)
        logger.std_info(f'[Parameters] Seed = {seed}')

        goals = []
        tasks = []

        if input_image_checkbox:
            if (current_tab == 'uov' or (
                    current_tab == 'ip' and mixing_image_prompt_and_vary_upscale)) \
                    and uov_method != flags.disabled and uov_input_image is not None:
                uov_input_image = HWC3(uov_input_image)
                if 'vary' in uov_method:
                    goals.append('vary')
                elif 'upscale' in uov_method:
                    goals.append('upscale')
                    if 'fast' in uov_method:
                        skip_prompt_processing = True
                    else:
                        steps = performance_selection.steps_uov()

                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    config.downloading_upscale_model()
            if (current_tab == 'inpaint' or (
                    current_tab == 'ip' and mixing_image_prompt_and_inpaint)) \
                    and isinstance(inpaint_input_image, dict):
                inpaint_image = inpaint_input_image['image']
                inpaint_mask = inpaint_input_image['mask'][:, :, 0]

                if inpaint_mask_upload_checkbox:
                    if isinstance(inpaint_mask_image_upload, np.ndarray):
                        if inpaint_mask_image_upload.ndim == 3:
                            H, W, C = inpaint_image.shape
                            inpaint_mask_image_upload = resample_image(inpaint_mask_image_upload, width=W, height=H)
                            inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
                            inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
                            inpaint_mask = np.maximum(np.zeros(shape=(H, W), dtype=np.uint8), inpaint_mask_image_upload)

                if int(inpaint_erode_or_dilate) != 0:
                    inpaint_mask = erode_or_dilate(inpaint_mask, inpaint_erode_or_dilate)

                if invert_mask_checkbox:
                    inpaint_mask = 255 - inpaint_mask

                inpaint_image = HWC3(inpaint_image)
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    config.downloading_upscale_model()
                    if inpaint_parameterized:
                        progressbar(async_task, 1, 'Downloading inpainter ...')
                        inpaint_head_model_path, inpaint_patch_model_path = config.downloading_inpaint_models(
                            inpaint_engine)
                        base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                        logger.std_info(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                        if refiner_model_name == 'None':
                            use_synthetic_refiner = True
                            refiner_switch = 0.8
                    else:
                        inpaint_head_model_path, inpaint_patch_model_path = None, None
                        logger.std_info('[Inpaint] Parameterized inpaint is disabled.')
                    if inpaint_additional_prompt != '':
                        if prompt == '':
                            prompt = inpaint_additional_prompt
                        else:
                            prompt = inpaint_additional_prompt + '\n' + prompt
                    goals.append('inpaint')
            if current_tab == 'ip' or \
                    mixing_image_prompt_and_vary_upscale or \
                    mixing_image_prompt_and_inpaint:
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

        if overwrite_step > 0:
            steps = overwrite_step

        switch = int(round(steps * refiner_switch))

        if overwrite_switch > 0:
            switch = overwrite_switch

        if overwrite_width > 0:
            width = overwrite_width

        if overwrite_height > 0:
            height = overwrite_height

        logger.std_info(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        logger.std_info(f'[Parameters] Steps = {steps} - {switch}')

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
            lora_filenames = remove_performance_lora(config.lora_filenames, performance_selection)
            loras, prompt = parse_lora_references_from_prompt(prompt, loras, config.default_max_lora_number, lora_filenames=lora_filenames)
            loras += performance_loras

            pipeline.refresh_everything(
                refiner_model_name=refiner_model_name,
                base_model_name=base_model_name,
                loras=loras,
                base_model_additional_loras=base_model_additional_loras,
                use_synthetic_refiner=use_synthetic_refiner)

            pipeline.set_clip_skip(clip_skip)

            progressbar(async_task, 3, 'Processing prompts ...')
            tasks = []

            for i in range(image_number):
                if disable_seed_increment:
                    task_seed = seed % (constants.MAX_SEED + 1)
                else:
                    task_seed = (seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not

                task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future
                task_prompt = apply_wildcards(prompt, task_rng, i, read_wildcards_in_order)
                task_prompt = apply_arrays(task_prompt, i)
                task_negative_prompt = apply_wildcards(negative_prompt, task_rng, i, read_wildcards_in_order)
                task_extra_positive_prompts = [apply_wildcards(pmt, task_rng, i, read_wildcards_in_order) for pmt in
                                               extra_positive_prompts]
                task_extra_negative_prompts = [apply_wildcards(pmt, task_rng, i, read_wildcards_in_order) for pmt in
                                               extra_negative_prompts]

                positive_basic_workloads = []
                negative_basic_workloads = []

                task_styles = style_selections.copy()
                if use_style:
                    for index, style in enumerate(task_styles):
                        if style == random_style_name:
                            style = get_random_style(task_rng)
                            task_styles[index] = style
                        p, n = apply_style(style, positive=task_prompt)
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
                    styles=task_styles
                ))

            if use_expansion:
                for i, t in enumerate(tasks):
                    progressbar(async_task, 4, f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    logger.std_info(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            for i, t in enumerate(tasks):
                progressbar(async_task, 5, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    progressbar(async_task, 6, f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if len(goals) > 0:
            progressbar(async_task, 7, 'Image processing ...')

        if 'vary' in goals:
            if 'subtle' in uov_method:
                denoising_strength = 0.5
            if 'strong' in uov_method:
                denoising_strength = 0.85
            if overwrite_vary_strength > 0:
                denoising_strength = overwrite_vary_strength

            shape_ceil = get_image_shape_ceil(uov_input_image)
            if shape_ceil < 1024:
                logger.std_warn('[Vary] Image is resized because it is too small.')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                logger.std_warn('[Vary] Image is resized because it is too big.')
                shape_ceil = 2048

            uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 8, 'VAE encoding ...')

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
            logger.std_info(f'[Vary] Final resolution is {str((height, width))}.')

        if 'upscale' in goals:
            H, W, C = uov_input_image.shape
            progressbar(async_task, 9, f'Upscaling image from {str((H, W))} ...')
            uov_input_image = perform_upscale(uov_input_image)
            logger.std_info('[Upscale] Image upscale.')

            if upscale_value is not None and upscale_value > 1.0:
                f = upscale_value
            else:
                if '1.5x' in uov_method:
                    f = 1.5
                elif '2x' in uov_method:
                    f = 2.0
                else:
                    f = 1.0

            shape_ceil = get_shape_ceil(H * f, W * f)

            if shape_ceil < 1024:
                logger.std_info('[Upscale] Image is resized because it is too small.')
                uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
                shape_ceil = 1024
            else:
                uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)

            image_is_super_large = shape_ceil > 2800

            if 'fast' in uov_method:
                direct_return = True
            elif image_is_super_large:
                logger.std_info('[Upscale] Image is too large. Directly returned the SR image. '
                                'Usually directly return SR image at 4K resolution '
                                'yields better results than SDXL diffusion.')
                direct_return = True
            else:
                direct_return = False

            if direct_return:
                # d = [('Upscale (Fast)', '2x')]
                # log(uov_input_image, d, output_format=save_extension)
                if config.default_black_out_nsfw or black_out_nsfw:
                    uov_input_image = default_censor(uov_input_image)
                yield_result(async_task, uov_input_image, tasks, save_extension, False, False)
                return

            tiled = True
            denoising_strength = 0.382

            if overwrite_upscale_strength > 0:
                denoising_strength = overwrite_upscale_strength

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 10, 'VAE encoding ...')

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
            logger.std_info(f'[Upscale] Final resolution is {str((height, width))}.')

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
                inpaint_strength = 1.0
                inpaint_respective_field = 1.0

            denoising_strength = inpaint_strength

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=inpaint_respective_field
            )

            if debugging_inpaint_preprocessor:
                yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(), tasks,
                             black_out_nsfw)
                return

            progressbar(async_task, 11, 'VAE Inpaint encoding ...')

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
                progressbar(async_task, 12, 'VAE SD15 encoding ...')
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

            if not inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}

            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            logger.std_info(f'[Inpaint] Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            for task in cn_tasks[flags.cn_canny]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not skipping_cn_preprocessor:
                    cn_img = preprocessors.canny_pyramid(cn_img, canny_low_threshold, canny_high_threshold)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, tasks, save_extension, black_out_nsfw)
                    return
            for task in cn_tasks[flags.cn_cpds]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not skipping_cn_preprocessor:
                    cn_img = preprocessors.cpds(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, tasks, save_extension, black_out_nsfw)
                    return
            for task in cn_tasks[flags.cn_ip]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, tasks, save_extension, black_out_nsfw)
                    return
            for task in cn_tasks[flags.cn_ip_face]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                if not skipping_cn_preprocessor:
                    cn_img = face_crop.crop_image(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, tasks, save_extension, black_out_nsfw)
                    return

            all_ip_tasks = cn_tasks[flags.cn_ip] + cn_tasks[flags.cn_ip_face]

            if len(all_ip_tasks) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

        if freeu_enabled:
            logger.std_info('[Fooocus] FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                freeu_b1,
                freeu_b2,
                freeu_s1,
                freeu_s2
            )

        all_steps = steps * image_number

        logger.std_info(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        logger.std_info(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        logger.std_info(f'[Fooocus] Preparation time: {preparation_time:.2f} seconds')

        final_sampler_name = sampler_name
        final_scheduler_name = scheduler_name

        if scheduler_name in ['lcm', 'tcd']:
            final_scheduler_name = 'sgm_uniform'

            def patch_discrete(unet):
                return core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling=scheduler_name,
                    zsnr=False)[0]

            if pipeline.final_unet is not None:
                pipeline.final_unet = patch_discrete(pipeline.final_unet)
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = patch_discrete(pipeline.final_refiner_unet)
            logger.std_info(f'[Fooocus] Using {scheduler_name} scheduler.')
        elif scheduler_name == 'edm_playground_v2.5':
            final_scheduler_name = 'karras'

            def patch_edm(unet):
                return core.opModelSamplingContinuousEDM.patch(
                    unet,
                    sampling=scheduler_name,
                    sigma_max=120.0,
                    sigma_min=0.002)[0]

            if pipeline.final_unet is not None:
                pipeline.final_unet = patch_edm(pipeline.final_unet)
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = patch_edm(pipeline.final_refiner_unet)

            logger.std_info(f'[Fooocus] Using {scheduler_name} scheduler.')

        outputs.append(['preview', (13, 'Moving model to GPU ...', None)])

        def callback(step, x0, x, total_steps, y):
            """callback, used for progress and preview"""
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
                    refiner_swap_method=refiner_swap_method,
                    disable_preview=disable_preview
                )

                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                # Fooocus async_worker.py code end

                results += imgs
            except model_management.InterruptProcessingException as e:
                logger.std_warn("[Fooocus] User stopped")
                results = []
                results.append(ImageGenerationResult(
                    im=None, seed=task['task_seed'], finish_reason=GenerationFinishReason.user_cancel))
                async_task.set_result(results, True, str(e))
                break
            except Exception as e:
                logger.std_error(f'[Fooocus] Process error: {e}')
                logging.exception(e)
                results = []
                results.append(ImageGenerationResult(
                    im=None, seed=task['task_seed'], finish_reason=GenerationFinishReason.error))
                async_task.set_result(results, True, str(e))
                break

            execution_time = time.perf_counter() - execution_start_time
            logger.std_info(f'[Fooocus] Generating and saving time: {execution_time:.2f} seconds')

        if async_task.finish_with_error:
            worker_queue.finish_task(async_task.job_id)
            return async_task.task_result
        yield_result(None, results, tasks, save_extension, black_out_nsfw)
        return
    except Exception as e:
        logger.std_error(f'[Fooocus] Worker error: {e}')

        if not async_task.is_finished:
            async_task.set_result([], True, str(e))
            worker_queue.finish_task(async_task.job_id)
            logger.std_info(f"[Task Queue] Finish task with error, job_id={async_task.job_id}")
