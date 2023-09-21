import argparse
import os
import shutil

from fooocusapi.api import start_app
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import pygit2

from fooocus_api_version import version
from fooocusapi.repositories_versions import fooocus_version, fooocus_commit_hash, comfy_commit_hash

comfyui_name = 'ComfyUI-from-StabilityAI-Official'
fooocus_name = 'Fooocus'

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = modules_path
dir_repos = "repositories"

def git_clone(url, dir, name, hash=None):
    """
    This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) Repository.
    """
    try:
        try:
            repo = pygit2.Repository(dir)
            print(f'{name} exists.')
        except:
            if os.path.exists(dir):
                shutil.rmtree(dir, ignore_errors=True)
            os.makedirs(dir, exist_ok=True)
            repo = pygit2.clone_repository(url, dir)
            print(f'{name} cloned.')

        remote = repo.remotes['origin']
        remote.fetch()

        commit = repo.get(hash)

        repo.checkout_tree(commit, strategy=pygit2.GIT_CHECKOUT_FORCE)
        print(f'{name} checkout finished.')
    except Exception as e:
        print(f'Git clone failed for {name}: {str(e)}')

def repo_dir(name):
    """
    This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) Repository.
    """
    return os.path.join(script_path, dir_repos, name)

def download_repositories():
    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version}")
    print(f"Fooscus-API version: {version}")

    # Check and download ComfyUI
    comfy_repo = os.environ.get('COMFY_REPO', "https://github.com/lllyasviel/ComfyUI_2bc12d.git")
    git_clone(comfy_repo, repo_dir(comfyui_name), "Inference Engine", comfy_commit_hash)

    # Check and download Fooocus
    fooocus_repo = os.environ.get('FOOOCUS_REPO', 'https://github.com/lllyasviel/Fooocus')
    git_clone(fooocus_repo, repo_dir(fooocus_name), "Fooocus", fooocus_commit_hash)

def download_models():
    model_filenames = [
        ('sd_xl_base_1.0_0.9vae.safetensors',
        'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors'),
        ('sd_xl_refiner_1.0_0.9vae.safetensors',
        'https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors')
    ]

    lora_filenames = [
        ('sd_xl_offset_example-lora_1.0.safetensors',
        'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors')
    ]

    vae_approx_filenames = [
        ('xlvaeapp.pth',
        'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth')
    ]

    upscaler_filenames = [
        ('fooocus_upscaler_s409985e5.bin',
        'https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_upscaler_s409985e5.bin')
    ]

    from modules.model_loader import load_file_from_url
    from modules.path import modelfile_path, lorafile_path, vae_approx_path, upscale_models_path, fooocus_expansion_path

    for file_name, url in model_filenames:
        load_file_from_url(url=url, model_dir=modelfile_path, file_name=file_name)
    for file_name, url in lora_filenames:
        load_file_from_url(url=url, model_dir=lorafile_path, file_name=file_name)
    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=vae_approx_path, file_name=file_name)
    for file_name, url in upscaler_filenames:
        load_file_from_url(url=url, model_dir=upscale_models_path, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=fooocus_expansion_path,
        file_name='pytorch_model.bin'
    )

def prepare_environments(args):
    skip_sync_repo = False
    if args.sync_repo is not None:
        if args.sync_repo == 'only':
            print("Only download and sync depent repositories")
            download_repositories()
            models_path = os.path.join(script_path, dir_repos, fooocus_name, "models")
            print(f"Sync repositories successful. Now you can put model files in subdirectories of '{models_path}'")
            return
        elif args.sync_repo == 'skip':
            skip_sync_repo = True
        else:
            print(f"Invalid value for argument '--sync-repo', acceptable value are 'skip' and 'only'")
            exit(1)

    if not skip_sync_repo:
        download_repositories()

    # Add indent repositories to import path
    sys.path.append(os.path.join(script_path, dir_repos, comfyui_name))
    sys.path.append(os.path.join(script_path, dir_repos, fooocus_name))

    download_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888, help="Set the listen port")
    parser.add_argument("--host", type=str, default='127.0.0.1', help="Set the listen host")
    parser.add_argument("--log-level", type=str, default='info', help="Log info for Uvicorn")
    parser.add_argument("--sync-repo", default=None, help="Sync dependent git repositories to local, 'skip' for skip sync action, 'only' for only do the sync action and not launch app")

    args = parser.parse_args()
    prepare_environments(args)

    # Start api server
    start_app(args)