# -*- coding: utf-8 -*-

""" Entry for Fooocus API.

Use for starting Fooocus API.
    python main.py --help for more usage

@file: main.py
@author: Konie
@update: 2024-03-22 
"""
import argparse
import os
import re
import shutil
import sys
from threading import Thread

from fooocusapi.utils.logger import logger
from fooocusapi.utils.tools import (
    run_pip,
    check_torch_cuda,
    requirements_check
)
from fooocus_api_version import version

script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(script_path, 'repositories/Fooocus')

sys.path.append(script_path)
sys.path.append(module_path)


print('[System ARGV] ' + str(sys.argv))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

python = sys.executable
default_command_live = True
index_url = os.environ.get('INDEX_URL', "")
re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def install_dependents(skip: bool = False):
    """
    Check and install dependencies
    Args:
        skip: skip pip install
    """
    if skip:
        return

    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")

    # Check if you need pip install
    if not requirements_check():
        run_pip("install -r requirements.txt", "requirements")

    if not check_torch_cuda():
        run_pip(f"install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}",
                desc="torch")


def preload_pipeline():
    """Preload pipeline"""
    logger.std_info("[Fooocus-API] Preloading pipeline ...")
    import modules.default_pipeline as _


def prepare_environments(args) -> bool:
    """
    Prepare environments
    Args:
        args: command line arguments
    """
    if args.gpu_device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
        print("Set device to:", args.gpu_device_id)

    if args.base_url is None or len(args.base_url.strip()) == 0:
        host = args.host
        if host == '0.0.0.0':
            host = '127.0.0.1'
        args.base_url = f"http://{host}:{args.port}"

    sys.argv = [sys.argv[0]]

    # Remove and copy preset folder
    origin_preset_folder = os.path.abspath(os.path.join(module_path, 'presets'))
    preset_folder = os.path.abspath(os.path.join(script_path, 'presets'))
    if os.path.exists(preset_folder):
        shutil.rmtree(preset_folder)
    shutil.copytree(origin_preset_folder, preset_folder)

    from modules import config
    from fooocusapi import parameters
    from fooocusapi.utils.model_loader import download_models
    parameters.default_inpaint_engine_version = config.default_inpaint_engine_version
    parameters.default_styles = config.default_styles
    parameters.default_base_model_name = config.default_base_model_name
    parameters.default_refiner_model_name = config.default_refiner_model_name
    parameters.default_refiner_switch = config.default_refiner_switch
    parameters.default_loras = config.default_loras
    parameters.default_cfg_scale = config.default_cfg_scale
    parameters.default_prompt_negative = config.default_prompt_negative
    parameters.default_aspect_ratio = parameters.get_aspect_ratio_value(config.default_aspect_ratio)
    parameters.available_aspect_ratios = [parameters.get_aspect_ratio_value(a) for a in config.available_aspect_ratios]

    download_models()

    # Init task queue
    from fooocusapi import worker
    from fooocusapi.task_queue import TaskQueue
    worker.worker_queue = TaskQueue(
        queue_size=args.queue_size,
        history_size=args.queue_history,
        webhook_url=args.webhook_url,
        persistent=args.persistent)

    logger.std_info(f"[Fooocus-API] Task queue size: {args.queue_size}")
    logger.std_info(f"[Fooocus-API] Queue history size: {args.queue_history}")
    logger.std_info(f"[Fooocus-API] Webhook url: {args.webhook_url}")

    return True


def pre_setup(disable_image_log: bool = False,
              skip_pip=False,
              load_all_models: bool = False,
              preload_pipeline: bool = False,
              always_gpu: bool = False,
              all_in_fp16: bool = False,
              preset: str | None = None):
    class Args(object):
        host = '127.0.0.1'
        port = 8888
        base_url = None
        disable_image_log = False
        skip_pip = False
        preload_pipeline = False
        queue_size = 100
        queue_history = 0
        preset = None
        webhook_url = None
        persistent = False
        always_gpu = False
        all_in_fp16 = False
        gpu_device_id = None
        apikey = None

    print("[Pre Setup] Prepare environments")

    args = Args()
    args.disable_image_log = disable_image_log
    args.skip_pip = skip_pip
    args.preload_pipeline = preload_pipeline
    args.always_gpu = always_gpu
    args.all_in_fp16 = all_in_fp16
    args.preset = preset

    sys.argv = [sys.argv[0]]
    if args.preset is not None:
        sys.argv.append('--preset')
        sys.argv.append(args.preset)

    if args.disable_image_log:
        sys.argv.append('--disable-image-log')

    install_dependents(args.skip_pip)

    import fooocusapi.args as _
    prepare_environments(args)

    if load_all_models:
        from modules import config
        from fooocusapi.parameters import default_inpaint_engine_version
        config.downloading_upscale_model()
        config.downloading_inpaint_models(default_inpaint_engine_version)
        config.downloading_controlnet_canny()
        config.downloading_controlnet_cpds()
        config.downloading_ip_adapters('ip')
        config.downloading_ip_adapters('face')
    print("[Pre Setup] Finished")


if __name__ == "__main__":
    logger.std_info(f"[Fooocus API] Python {sys.version}")
    logger.std_info(f"[Fooocus API] Fooocus API version: {version}")

    from fooocusapi.base_args import add_base_args

    parser = argparse.ArgumentParser()
    add_base_args(parser, True)

    args, _ = parser.parse_known_args()
    install_dependents(skip=args.skip_pip)

    from fooocusapi.args import args

    if prepare_environments(args):
        sys.argv = [sys.argv[0]]

        # Load pipeline in new thread
        preload_pipeline_thread = Thread(target=preload_pipeline, daemon=True)
        preload_pipeline_thread.start()

        # Start task schedule thread
        from fooocusapi.worker import task_schedule_loop
        task_schedule_thread = Thread(target=task_schedule_loop, daemon=True)
        task_schedule_thread.start()

        # Start api server
        from fooocusapi.api import start_app

        start_app(args)
