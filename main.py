import argparse
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec

from fooocus_api_version import version
from fooocusapi.repositories_versions import (comfy_commit_hash,
                                              fooocus_commit_hash,
                                              fooocus_version)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

python = sys.executable
default_command_live = True
index_url = os.environ.get('INDEX_URL', "")

comfyui_name = 'ComfyUI-from-StabilityAI-Official'
fooocus_name = 'Fooocus'

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = modules_path
dir_repos = "repositories"


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def onerror(func, path, exc_info):
    import stat
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise 'Failed to invoke "shutil.rmtree", git management failed.'


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def git_clone(url, dir, name, hash=None):
    import pygit2

    try:
        try:
            repo = pygit2.Repository(dir)
            remote_url = repo.remotes['origin'].url
            if remote_url != url:
                print(f'{name} exists but remote URL will be updated.')
                del repo
                raise url
            else:
                print(f'{name} exists and URL is correct.')
        except:
            if os.path.isdir(dir) or os.path.exists(dir):
                shutil.rmtree(dir, onerror=onerror)
            os.makedirs(dir, exist_ok=True)
            repo = pygit2.clone_repository(url, dir)
            print(f'{name} cloned from {url}.')

        remote = repo.remotes['origin']
        remote.fetch()

        commit = repo.get(hash)

        repo.checkout_tree(commit, strategy=pygit2.GIT_CHECKOUT_FORCE)
        repo.set_head(commit.id)

        print(f'{name} checkout finished for {hash}.')
    except Exception as e:
        print(f'Git clone failed for {name}: {str(e)}')


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def run_pip(command, desc=None, live=default_command_live):
    try:
        index_url_line = f' --index-url {index_url}' if index_url != '' else ''
        return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}",
                   errdesc=f"Couldn't install {desc}", live=live)
    except Exception as e:
        print(e)
        print(f'CMD Failed {desc}: {command}')
        return None


def download_repositories():
    import pygit2

    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

    http_proxy = os.environ.get('HTTP_PROXY')
    https_proxy = os.environ.get('HTTPS_PROXY')
    
    if http_proxy != None:
        print(f"Using http proxy for git clone: {http_proxy}")
        os.environ['http_proxy'] = http_proxy

    if https_proxy != None:
        print(f"Using https proxy for git clone: {https_proxy}")
        os.environ['https_proxy'] = https_proxy

    # Check and download ComfyUI
    comfy_repo = os.environ.get(
        'COMFY_REPO', "https://github.com/comfyanonymous/ComfyUI")
    git_clone(comfy_repo, repo_dir(comfyui_name),
              "Inference Engine", comfy_commit_hash)
    
    # Check and download Fooocus
    fooocus_repo = os.environ.get(
        'FOOOCUS_REPO', 'https://github.com/lllyasviel/Fooocus')
    git_clone(fooocus_repo, repo_dir(fooocus_name),
              "Fooocus", fooocus_commit_hash)
    

def is_installed(package):
    try:
        spec = find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


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

    from modules.model_loader import load_file_from_url
    from modules.path import (fooocus_expansion_path, lorafile_path,
                              modelfile_path, upscale_models_path,
                              vae_approx_path)

    for file_name, url in model_filenames:
        load_file_from_url(url=url, model_dir=modelfile_path,
                           file_name=file_name)
    for file_name, url in lora_filenames:
        load_file_from_url(url=url, model_dir=lorafile_path,
                           file_name=file_name)
    for file_name, url in vae_approx_filenames:
        load_file_from_url(
            url=url, model_dir=vae_approx_path, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=fooocus_expansion_path,
        file_name='pytorch_model.bin'
    )


def run_pip_install():
    print("Run pip install")
    run_pip("install -r requirements.txt", "requirements")
    run_pip("install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118", "torch")
    run_pip("install xformers", "xformers")


def prepare_environments(args) -> bool:
    # Check if need pip install
    if not is_installed('xformers'):
        run_pip_install()

    skip_sync_repo = False
    if args.sync_repo is not None:
        if args.sync_repo == 'only':
            print("Only download and sync depent repositories")
            download_repositories()
            models_path = os.path.join(
                script_path, dir_repos, fooocus_name, "models")
            print(
                f"Sync repositories successful. Now you can put model files in subdirectories of '{models_path}'")
            return False
        elif args.sync_repo == 'skip':
            skip_sync_repo = True
        else:
            print(
                f"Invalid value for argument '--sync-repo', acceptable value are 'skip' and 'only'")
            exit(1)

    if not skip_sync_repo:
        download_repositories()

    # Add indent repositories to import path
    sys.path.append(os.path.join(script_path, dir_repos, comfyui_name))
    sys.path.append(os.path.join(script_path, dir_repos, fooocus_name))

    download_models()
    return True


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def ini_comfy_args():
    argv = sys.argv
    sys.argv = [sys.argv[0]]

    from comfy.cli_args import args as comfy_args
    comfy_args.disable_cuda_malloc = True
    comfy_args.auto_launch = False

    sys.argv = argv


if __name__ == "__main__":
    print(f"Python {sys.version}")
    print(f"Fooocus-API version: {version}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888,
                        help="Set the listen port")
    parser.add_argument("--host", type=str,
                        default='127.0.0.1', help="Set the listen host")
    parser.add_argument("--log-level", type=str,
                        default='info', help="Log info for Uvicorn")
    parser.add_argument("--sync-repo", default=None,
                        help="Sync dependent git repositories to local, 'skip' for skip sync action, 'only' for only do the sync action and not launch app")

    args = parser.parse_args()

    if prepare_environments(args):
        ini_comfy_args()

        # Start api server
        from fooocusapi.api import start_app
        start_app(args)
