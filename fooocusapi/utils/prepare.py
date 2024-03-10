"""Prepare the environment"""
# pylint: disable=line-too-long
# pylint: disable=broad-exception-caught
# pylint: disable=import-outside-toplevel
import os
import sys
import shutil
import re
import importlib.metadata
import packaging.version

from modules.model_loader import load_file_from_url

from fooocusapi.utils.tools import (is_installed,
                                    run_pip,
                                    check_torch_cuda)


PATTERN = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def requirements_check(requirements_file: str, pattern: re.Pattern) -> bool:
    """
    Check if the requirements file is satisfied
    Args:
        requirements_file: Path to the requirements file
        pattern: Pattern to match the requirements
    Returns:
        Whether the requirements file is satisfied
    """
    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            if line.strip() == "":
                continue

            m = re.match(pattern, line)
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


def install_dependents(args, pattern: re.Pattern = PATTERN):
    """
    Install the dependencies
    Args:
        args: Arguments
        pattern: Pattern to match the requirements
    """
    if not args.skip_pip:
        torch_index_url = os.environ.get('TORCH_INDEX_URL',
                                         "https://download.pytorch.org/whl/cu121")

        # Check if you need pip install
        requirements_file = 'requirements.txt'
        if not requirements_check(requirements_file=requirements_file,
                                  pattern=pattern):
            command = f"install -r \"{requirements_file}\""
            run_pip(command=command, desc="Installing dependencies")

        if not is_installed("torch") or not is_installed("torchvision"):
            run_pip(command=f"install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}",
                    desc="Pytorch installing ..")

        if args.persistent and not is_installed("sqlalchemy"):
            run_pip("install sqlalchemy==2.0.25", "sqlalchemy")


def download_models():
    """
    Downloads the models.
    """
    from modules.config import (path_checkpoints as modelfile_path,
                            path_loras as lorafile_path,
                            path_vae_approx as vae_approx_path,
                            path_fooocus_expansion as fooocus_expansion_path,
                            checkpoint_downloads,
                            path_embeddings as embeddings_path,
                            embeddings_downloads, lora_downloads)
    uri = 'https://huggingface.co/lllyasviel/misc/resolve/main/'
    vae_approx_filenames = [
        ('xlvaeapp.pth', f'{uri}xlvaeapp.pth'),
        ('vaeapp_sd15.pth', f'{uri}vaeapp_sd15.pt'),
        ('xl-to-v1_interposer-v3.1.safetensors', f'{uri}xl-to-v1_interposer-v3.1.safetensors')
    ]
    fooocus_expansion_files = [
        ('pytorch_model.bin', f'{uri}fooocus_expansion.bin')
    ]

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


def prepare_environments(args, module_path, script_path) -> bool:
    """
    Prepare the environments
    Args:
        args: Arguments
        module_path: Module path, where the Fooocus is located
        script_path: Script path, where the app root is located
    """
    if not check_torch_cuda():
        print("Torch or Cuda not available, Application may not work.")

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

    download_models()
    return True


def preplaod_pipeline():
    "Preload pipeline"
    print("Preload pipeline")
    import modules.default_pipeline as _
