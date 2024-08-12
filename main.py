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
from fooocusapi.utils.tools import run_pip, check_torch_cuda, requirements_check
from fooocus_api_version import version

script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(script_path, "repositories/Fooocus")

sys.path.append(script_path)
sys.path.append(module_path)


logger.std_info("[System ARGV] " + str(sys.argv))

try:
    index = sys.argv.index('--gpu-device-id')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[index+1])
    logger.std_info(f"[Fooocus] Set device to: {str(sys.argv[index+1])}")
except ValueError:
    pass

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

python = sys.executable
default_command_live = True
index_url = os.environ.get("INDEX_URL", "")
re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


def install_dependents(skip: bool = False):
    """
    Check and install dependencies
    Args:
        skip: skip pip install
    """
    if skip:
        return

    torch_index_url = os.environ.get(
        "TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121"
    )

    # Check if you need pip install
    if not requirements_check():
        run_pip("install -r requirements.txt", "requirements")

    if not check_torch_cuda():
        run_pip(
            f"install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}",
            desc="torch",
        )


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

    if args.base_url is None or len(args.base_url.strip()) == 0:
        host = args.host
        if host == "0.0.0.0":
            host = "127.0.0.1"
        args.base_url = f"http://{host}:{args.port}"

    sys.argv = [sys.argv[0]]

    # Remove and copy preset folder
    origin_preset_folder = os.path.abspath(os.path.join(module_path, "presets"))
    preset_folder = os.path.abspath(os.path.join(script_path, "presets"))
    if os.path.exists(preset_folder):
        shutil.rmtree(preset_folder)
    shutil.copytree(origin_preset_folder, preset_folder)

    from modules import config
    from fooocusapi.configs import default
    from fooocusapi.utils.model_loader import download_models

    default.default_inpaint_engine_version = config.default_inpaint_engine_version
    default.default_styles = config.default_styles
    default.default_base_model_name = config.default_base_model_name
    default.default_refiner_model_name = config.default_refiner_model_name
    default.default_refiner_switch = config.default_refiner_switch
    default.default_loras = config.default_loras
    default.default_cfg_scale = config.default_cfg_scale
    default.default_prompt_negative = config.default_prompt_negative
    default.default_aspect_ratio = default.get_aspect_ratio_value(
        config.default_aspect_ratio
    )
    default.available_aspect_ratios = [
        default.get_aspect_ratio_value(a) for a in config.available_aspect_ratios
    ]

    if not args.disable_preset_download:
        download_models()

    # Init task queue
    from fooocusapi import worker
    from fooocusapi.task_queue import TaskQueue

    worker.worker_queue = TaskQueue(
        queue_size=args.queue_size,
        history_size=args.queue_history,
        webhook_url=args.webhook_url,
        persistent=args.persistent,
    )

    logger.std_info(f"[Fooocus-API] Task queue size: {args.queue_size}")
    logger.std_info(f"[Fooocus-API] Queue history size: {args.queue_history}")
    logger.std_info(f"[Fooocus-API] Webhook url: {args.webhook_url}")

    return True


def pre_setup():
    """
    Pre setup, for replicate
    """
    class Args(object):
        """
        Arguments object
        """
        host = "127.0.0.1"
        port = 8888
        base_url = None
        sync_repo = "skip"
        disable_image_log = True
        skip_pip = True
        preload_pipeline = True
        queue_size = 100
        queue_history = 0
        preset = "default"
        webhook_url = None
        persistent = False
        always_gpu = False
        all_in_fp16 = False
        gpu_device_id = None
        apikey = None

    print("[Pre Setup] Prepare environments")

    arguments = Args()
    sys.argv = [sys.argv[0]]
    sys.argv.append("--disable-image-log")

    install_dependents(arguments.skip_pip)

    prepare_environments(arguments)

    # Start task schedule thread
    from fooocusapi.worker import task_schedule_loop

    task_thread = Thread(target=task_schedule_loop, daemon=True)
    task_thread.start()

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
