import random
import numpy as np

from modules import constants, flags
from modules.util import (HWC3, resample_image, erode_or_dilate, remove_empty_str)
from modules.sdxl_styles import fooocus_expansion, apply_wildcards, apply_style
from modules.config import (
    downloading_upscale_model,
    downloading_inpaint_models,
    downloading_controlnet_cpds,
    downloading_controlnet_canny,
    downloading_ip_adapters
)
from fooocusapi.utils.logger import default_logger


def check_aspect_ratios(aspect_ratios: str) -> tuple[int, int]:
    """
    Check if the aspect ratio format is valid and return the width and height.
    Args:
        aspect_ratios: str - The aspect ratio in the format 'width*height'.
    Returns:
        tuple[int, int] - The width and height as integers.
    """
    try:
        width, height = aspect_ratios.split('*')[:2]
        return int(width), int(height)
    except ValueError as ve:
        raise ValueError("Invalid aspect ratio format. Please use 'width*height'.") from ve


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


def uov_steps_based_on_performance(performance_selection: str) -> int:
    """
    Set the number of steps based on the performance selection.
    Args:
        performance_selection: performance selection string
    Returns:
    """
    if performance_selection == 'Speed':
        return 18
    elif performance_selection == 'Quality':
        return 36
    elif performance_selection == 'Extreme Speed':
        return 8
    else:
        return 18


def steps_based_on_performance(performance_selection: str) -> int:
    """
    Set the number of steps based on the performance selection.
    Args:
        performance_selection:
    Returns:
    """
    if performance_selection == 'Speed':
        return 30
    elif performance_selection == 'Quality':
        return 60
    elif performance_selection == 'Extreme Speed':
        return 8
    else:
        return 30


def overwrite_by_ap(steps: int, switch: int,
                    width: int, height: int, ap: object) -> tuple:
    """
    Overwrite the parameters with the values from the AP.
    Args:
        steps: steps
        switch: switch
        width: width
        height: height
        ap: advanced parameters object
    Returns:
        tuple of steps, switch, width, height
    """
    if ap.overwrite_step > 0:
        steps = ap.overwrite_step
    if ap.overwrite_switch > 0:
        switch = ap.overwrite_switch
    if ap.overwrite_width > 0:
        width = ap.overwrite_width
    if ap.overwrite_height > 0:
        height = ap.overwrite_height
    return steps, switch, width, height


def process_inpaint_input_image(inpaint_input_image: dict | None) -> dict | bool:
    """
    Process the inpaint image parameters.
    Args:
        inpaint_input_image: inpaint image parameters
    Returns:
    """
    if inpaint_input_image is None or inpaint_input_image['image'] is None:
        return False
    else:
        inpaint_image_size = inpaint_input_image["image"].shape[:2]
        if inpaint_input_image['mask'] is None:
            inpaint_input_image['mask'] = np.zeros(inpaint_image_size, dtype=np.uint8)
        inpaint_input_image['mask'] = HWC3(inpaint_input_image['mask'])
        return inpaint_input_image


def expansion_style(style_selections: list[str]) -> tuple[bool, bool]:
    """
    Determine whether to use expansion and style.
    Args:
        style_selections: style selection list
    Returns:
        tuple[bool, bool]: use_expansion, use_style
    """
    if fooocus_expansion in style_selections:
        use_expansion = True
        style_selections.remove(fooocus_expansion)
    else:
        use_expansion = False

    use_style = len(style_selections) > 0
    return use_expansion, use_style


def determine_task_type(current_tab: str,
                        uov_method: str, uov_image,
                        inpaint_image: dict,
                        ap: object) -> str:
    """
    Determine the task type. One of the upscale or vary, inpaint, and image prompt
    Args:
        current_tab: current tab
        uov_method: upscale or vary
        uov_image: uov input image
        inpaint_image: inpaint input image
        ap: advanced params
    Returns: one of 'uov', 'inpaint', 'ip' or 'none'
    """
    if (current_tab == 'uov' or (
            current_tab == 'ip' and ap.mixing_image_prompt_and_vary_upscale)) and \
            uov_method != flags.disabled and \
            uov_image is not None:
        return 'uov'
    if (current_tab == 'inpaint' or (
            current_tab == 'ip' and ap.mixing_image_prompt_and_inpaint)) \
            and isinstance(inpaint_image, dict):
        return 'inpaint'
    if current_tab == 'ip' or \
            ap.mixing_image_prompt_and_inpaint or \
            ap.mixing_image_prompt_and_vary_upscale:
        return 'ip'
    return 'none'


def process_uov(uov_method: bool, skip_prompt_processing: bool,
                performance_selection: str, steps: int) -> tuple:
    """
    Process upscale or vary
    Args:
        uov_method: upscale or vary
        skip_prompt_processing: skip prompt processing
        performance_selection: performance_selection
        steps: steps
    Returns:
        tuple of goals, skip_prompt_processing, steps
    """
    goals = []
    if 'vary' in uov_method:
        goals.append('vary')
    elif 'upscale' in uov_method:
        goals.append('upscale')
        if 'fast' in uov_method:
            skip_prompt_processing = True
        else:
            steps = uov_steps_based_on_performance(performance_selection)
        downloading_upscale_model()
    return goals, skip_prompt_processing, steps


def process_inpaint(inpaint_input_image: dict, inpaint_mask_image_upload,
                    outpaint_selections: list, inpaint_parameterized: bool,
                    inpaint_head_model_path, inpaint_patch_model_path,
                    use_synthetic_refiner: bool, refiner_switch: float,
                    base_model_additional_loras: list, refiner_model_name: str,
                    inpaint_additional_prompt: str, prompt: str,
                    ap: object) -> tuple:
    goals = []
    inpaint_image = inpaint_input_image['image']
    inpaint_mask = inpaint_input_image['mask'][:, :, 0]

    if ap.inpaint_mask_upload_checkbox:
        if isinstance(inpaint_mask_image_upload, np.ndarray):
            if ap.ndim == 3:
                height, width, channel = inpaint_image.shape
                inpaint_mask_image_upload = resample_image(ap, width=width, height=height)
                inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
                inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
                inpaint_mask = inpaint_mask_image_upload
    if int(ap.inpaint_erode_or_dilate) != 0:
        inpaint_mask = erode_or_dilate(inpaint_mask, ap.inpaint_erode_or_dilate)

    if ap.invert_mask_checkbox:
        inpaint_mask = 255 - inpaint_mask

    inpaint_image = HWC3(inpaint_image)
    if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
            and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
        downloading_upscale_model()
        if inpaint_parameterized:
            inpaint_head_model_path, inpaint_patch_model_path = downloading_inpaint_models(
                ap.inpaint_engine)
            base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
            default_logger.std_info(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
            if refiner_model_name == 'None':
                use_synthetic_refiner = True
                refiner_switch = 0.5
        else:
            inpaint_head_model_path, inpaint_patch_model_path = None, None
            default_logger.std_info('[Inpaint] Parameterized inpaint is disabled.')
        if inpaint_additional_prompt != '':
            if prompt == '':
                prompt = inpaint_additional_prompt
            else:
                prompt = inpaint_additional_prompt + '\n' + prompt

            goals.append('inpaint')

    return inpaint_image, inpaint_mask, inpaint_head_model_path, inpaint_patch_model_path, \
        base_model_additional_loras, use_synthetic_refiner, refiner_switch, prompt


def process_image_prompt(cn_tasks: dict) -> tuple:
    """
    Process image prompt
    Args:
        cn_tasks: control net tasks
    Returns: tuple of controlnet_canny_path, controlnet_cpds_path, clip_vision_path,
        ip_negative_path, ip_adapter_path, ip_adapter_face_path
    """
    controlnet_canny_path = None
    controlnet_cpds_path = None
    clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

    if len(cn_tasks[flags.cn_canny]) > 0:
        controlnet_canny_path = downloading_controlnet_canny()
    if len(cn_tasks[flags.cn_cpds]) > 0:
        controlnet_cpds_path = downloading_controlnet_cpds()
    if len(cn_tasks[flags.cn_ip]) > 0:
        clip_vision_path, ip_negative_path, ip_adapter_path = downloading_ip_adapters('ip')
    if len(cn_tasks[flags.cn_ip_face]) > 0:
        clip_vision_path, ip_negative_path, ip_adapter_face_path = downloading_ip_adapters('face')

    return controlnet_canny_path, controlnet_cpds_path, clip_vision_path, \
        ip_negative_path, ip_adapter_path, ip_adapter_face_path


def add_tasks(image_number: int, prompt: str, negative_prompt: str, seed: int,
              extra_positive_prompts: str, extra_negative_prompts: str,
              use_style: bool, style_selections: list) -> list[dict]:
    """
    Add tasks to the list.
    Args:
        image_number:
        prompt:
        negative_prompt:
        seed:
        extra_positive_prompts:
        extra_negative_prompts:
        use_style:
        style_selections:
    Returns: tasks list
    """
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
    return tasks
