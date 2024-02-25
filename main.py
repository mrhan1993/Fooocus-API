import argparse
import os
import re
import shutil
import subprocess
import sys

from importlib.util import find_spec
from threading import Thread

import importlib.metadata
import packaging.version
from fooocus_api_version import version


script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(script_path, 'repositories/Fooocus')

sys.path.append(script_path)
sys.path.append(module_path)


print('[System ARGV] ' + str(sys.argv))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


PYTHON_EXEC = sys.executable
INDEX_URL = os.environ.get('INDEX_URL', "")
RE_REQUIREMENTS = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def run_command(command: str,
                desc: str = None,
                error_desc: str = None,
                custom_env: str = None,
                live: bool = True) -> str:
    """
    :param command: Command to run.
    :param desc: Description of the command.
    :param error_desc: Description of the error.
    :param custom_env: Custom environment variables.
    :param live: Whether to print the command.
    :return: Output of the command.
    """
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore'
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{error_desc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return result.stdout or ""


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def run_pip(command, desc=None, live=True) -> str | None:
    """
    Runs a pip command.
    :param command: The command to run.
    :param desc: The description of the command.
    :param live: Whether to print the command output.
    :return: None
    """
    try:
        index_url_line = f' --index-url {INDEX_URL}' if INDEX_URL != '' else ''
        return run_command(f'"{PYTHON_EXEC}" -m pip {command} --prefer-binary{index_url_line}',
                           desc=f"Installing {desc}",
                           error_desc=f"Couldn't install {desc}",
                           live=live)
    except Exception as e:
        print(e)
        print(f'CMD Failed {desc}: {command}')
        return None


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def requirements_check(requirements_file: str) -> bool:
    """
    Does a simple parse of a requirements.txt file to determine if all requirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    :param requirements_file: The requirements file to parse.
    :return: True if all requirements are installed, False otherwise.
    """

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            if line.strip() == "":
                continue

            m = re.match(RE_REQUIREMENTS, line)
            if m is None:
                return False

            package = m.group(1).strip()
            version_required = (m.group(2) or "").strip()

            if version_required == "":
                continue

            try:
                version_installed = importlib.metadata.version(package)
            except Exception:
                return False

            if packaging.version.parse(version_required) != packaging.version.parse(version_installed):
                return False

    return True


def is_installed(package: str) -> bool:
    """
    Checks if a package is installed.
    :param package: The package to check.
    :return: True if installed, False otherwise.
    """
    try:
        spec = find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def download_models():
    """
    Downloads the models.
    """
    uri = 'https://huggingface.co/lllyasviel/misc/resolve/main/'
    vae_approx_filenames = [
        ('xlvaeapp.pth', f'{uri}xlvaeapp.pth'),
        ('vaeapp_sd15.pth', f'{uri}vaeapp_sd15.pt'),
        ('xl-to-v1_interposer-v3.1.safetensors', f'{uri}xl-to-v1_interposer-v3.1.safetensors')
    ]
    fooocus_expansion_files = [
        ('pytorch_model.bin', f'{uri}fooocus_expansion.bin')
    ]

    from modules.model_loader import load_file_from_url
    from modules.config import (path_checkpoints as modelfile_path,
                                path_loras as lorafile_path,
                                path_vae_approx as vae_approx_path,
                                path_fooocus_expansion as fooocus_expansion_path,
                                checkpoint_downloads,
                                path_embeddings as embeddings_path,
                                embeddings_downloads, lora_downloads)

    for file_name, url in checkpoint_downloads.items():
        load_file_from_url(url=url, model_dir=modelfile_path, file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=embeddings_path, file_name=file_name)
    for file_name, url in lora_downloads.items():
        load_file_from_url(url=url, model_dir=lorafile_path, file_name=file_name)
    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=vae_approx_path, file_name=file_name)
    for file_name, url in fooocus_expansion_files:
        load_file_from_url(url=url, model_dir=fooocus_expansion_path, file_name=file_name)


def install_dependents(args):
    """
    Install dependencies.
    :param args: The arguments.
    """
    if not args.skip_pip:
        torch_index_url = os.environ.get('TORCH_INDEX_URL',
                                         "https://download.pytorch.org/whl/cu121")

        # Check if you need pip install
        requirements_file = 'requirements.txt'
        if not requirements_check(requirements_file):
            run_pip(f"install -r \"{requirements_file}\"", "requirements")

        if not is_installed("torch") or not is_installed("torchvision"):
            print(f"torch_index_url: {torch_index_url}")
            run_pip(f"install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}",
                    "torch")

        if args.persistent and not is_installed("sqlalchemy"):
            run_pip("install sqlalchemy==2.0.25", "sqlalchemy")


def prepare_environments(args) -> bool:
    """
    Prepare the environments.
    :param args: The arguments.
    :return: True if successful, False otherwise.
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

    if args.preset is not None:
        # Remove and copy preset folder
        origin_preset_folder = os.path.abspath(os.path.join(module_path, 'presets'))
        preset_folder = os.path.abspath(os.path.join(script_path, 'presets'))
        if os.path.exists(preset_folder):
            shutil.rmtree(preset_folder)
        shutil.copytree(origin_preset_folder, preset_folder)

    import modules.config as config
    import fooocusapi.parameters as parameters
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
    import fooocusapi.worker as worker
    from fooocusapi.task_queue import TaskQueue
    worker.worker_queue = TaskQueue(queue_size=args.queue_size,
                                    hisotry_size=args.queue_history,
                                    webhook_url=args.webhook_url,
                                    persistent=args.persistent)
    print(f"[Fooocus-API] Task queue size: {args.queue_size}")
    print(f"[Fooocus-API] Task queue history size: {args.queue_history}")
    print(f"[Fooocus-API] Task queue webhook url: {args.webhook_url}")

    return True


def pre_setup(disable_image_log: bool = False, skip_pip=False,
              load_all_models: bool = False, preload_pipeline: bool = False,
              always_gpu: bool = False, all_in_fp16: bool = False, preset: str | None = None):
    """
    Prepare environments for replicate.
    :param disable_image_log: Disable image log.
    :param skip_pip: Skip pip install.
    :param load_all_models: Load all models.
    :param preload_pipeline: Preload pipeline.
    :param always_gpu: Always use GPU.
    :param all_in_fp16: All in fp16.
    :param preset: The preset.
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
    prepare_environments(args)

    if load_all_models:
        import modules.config as config
        from fooocusapi.parameters import default_inpaint_engine_version
        config.downloading_upscale_model()
        config.downloading_inpaint_models(default_inpaint_engine_version)
        config.downloading_controlnet_canny()
        config.downloading_controlnet_cpds()
        config.downloading_ip_adapters()
    print("[Pre Setup] Finished")


def preplaod_pipeline():
    "Preload pipeline"
    print("Preload pipeline")
    import modules.default_pipeline as _


if __name__ == "__main__":
    print(f"Python {sys.version}")
    print(f"Fooocus-API version: {version}")

    from fooocusapi.base_args import add_base_args

    parser = argparse.ArgumentParser()
    add_base_args(parser, True)

    args, _ = parser.parse_known_args()
    install_dependents(args)

    from fooocusapi.args import args

    if prepare_environments(args):
        sys.argv = [sys.argv[0]]

        # Load pipeline in new thread
        preload_pipeline_thread = Thread(target=preplaod_pipeline, daemon=True)
        preload_pipeline_thread.start()

        # Start task schedule thread
        from fooocusapi.worker import task_schedule_loop
        task_schedule_thread = Thread(target=task_schedule_loop, daemon=True)
        task_schedule_thread.start()

        # Start api server
        from fooocusapi.api import start_app

        start_app(args)
