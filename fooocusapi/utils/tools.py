# -*- coding: utf-8 -*-

""" Some tools

@file: tools.py
@author: Konie
@update: 2024-03-22
"""
# pylint: disable=line-too-long
# pylint: disable=broad-exception-caught
import os
import sys
import re
import subprocess
from importlib.util import find_spec
from importlib import metadata
from packaging import version


PYTHON_EXEC = sys.executable
INDEX_URL = os.environ.get('INDEX_URL', "")
PATTERN = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")


# This function was copied from [Fooocus](https://github.com/lllyasviel/Fooocus) repository.
def run_command(command: str,
                desc: str = None,
                error_desc: str = None,
                custom_env: str = None,
                live: bool = True) -> str:
    """
    Run a command and return the output
    Args:
        command: Command to run
        desc: Description of the command
        error_desc: Description of the error
        custom_env: Custom environment variables
        live: Whether to print the output
    Returns:
        The output of the command
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

    result = subprocess.run(check=False, **run_kwargs)

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
def run_pip(command, desc=None, live=True):
    """
    Run a pip command
    Args:
        command: Command to run
        desc: Description of the command
        live: Whether to print the output
    Returns:
        The output of the command
    """
    try:
        index_url_line = f' --index-url {INDEX_URL}' if INDEX_URL != '' else ''
        return run_command(
            command=f'"{PYTHON_EXEC}" -m pip {command} --prefer-binary{index_url_line}',
            desc=f"Installing {desc}",
            error_desc=f"Couldn't install {desc}",
            live=live
        )
    except Exception as e:
        print(f'CMD Failed {command}: {e}')
        return None


def is_installed(package: str) -> bool:
    """
    Check if a package is installed
    Args:
        package: Package name
    Returns:
        Whether the package is installed
    """
    try:
        spec = find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def check_torch_cuda() -> bool:
    """
    Check if torch and CUDA is available
    Returns:
        Whether CUDA is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def requirements_check(requirements_file: str = 'requirements.txt',
                       pattern: re.Pattern = PATTERN) -> bool:
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
                version_installed = metadata.version(package)
            except Exception:
                return False

            if version.parse(version_required) != version.parse(version_installed):
                return False

    return True
