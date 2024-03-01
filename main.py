"""Application startup file"""
# pylint: disable=import-outside-toplevel
import argparse
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(script_path, 'repositories/Fooocus')

sys.path.append(script_path)
sys.path.append(module_path)

from threading import Thread

from fooocusapi.worker import task_schedule_loop

from fooocusapi.utils.prepare import (prepare_environments,
                                      install_dependents,
                                      preplaod_pipeline)
from fooocus_api_version import version





print('[System ARGV] ' + str(sys.argv))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def pre_setup(disable_image_log: bool = False, skip_pip=False,
              load_all_models: bool = False, preload_pipeline: bool = False,
              always_gpu: bool = False, all_in_fp16: bool = False, preset: str | None = None):
    """
    Prepare environments for replicate.
    Args:
        disable_image_log: Disable image log.
        skip_pip: Skip pip install.
        load_all_models: Load all models.
        preload_pipeline: Preload pipeline.
        always_gpu: Always use GPU.
        all_in_fp16: All in fp16.
        preset: The preset.
    """
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

    install_dependents(args)
    preplaod_pipeline()
    prepare_environments(args, module_path, script_path)

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
    from fooocusapi.base_args import add_base_args
    from fooocusapi.api import start_app
    
    print(f"Python {sys.version}")
    print(f"Fooocus-API version: {version}")
    parser = argparse.ArgumentParser()
    add_base_args(parser, True)

    args, _ = parser.parse_known_args()
    install_dependents(args)

    from fooocusapi.args import args
    if prepare_environments(args, module_path, script_path):
        sys.argv = [sys.argv[0]]

        # Load pipeline in new thread
        preload_pipeline_thread = Thread(target=preplaod_pipeline, daemon=True)
        preload_pipeline_thread.start()

        # Start task schedule thread
        task_schedule_thread = Thread(target=task_schedule_loop, daemon=True)
        task_schedule_thread.start()

        # Start api server
        start_app(args)
