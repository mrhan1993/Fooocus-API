# -*- coding: utf-8 -*-

"""
Download models from url

@file: model_loader.py
@author: Konie
@update: 2024-03-22 
"""
from modules.model_loader import load_file_from_url


def download_models():
    """
    Download models from config
    """
    vae_approx_filenames = [
        ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
        ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
        ('xl-to-v1_interposer-v3.1.safetensors', 'https://huggingface.co/lllyasviel/misc/resolve/main/xl-to-v1_interposer-v3.1.safetensors')
    ]

    from modules.config import (
        paths_checkpoints as modelfile_path,
        paths_loras as lorafile_path,
        path_vae_approx as vae_approx_path,
        path_fooocus_expansion as fooocus_expansion_path,
        path_embeddings as embeddings_path,
        checkpoint_downloads,
        embeddings_downloads,
        lora_downloads)

    for file_name, url in checkpoint_downloads.items():
        load_file_from_url(url=url, model_dir=modelfile_path[0], file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=embeddings_path, file_name=file_name)
    for file_name, url in lora_downloads.items():
        load_file_from_url(url=url, model_dir=lorafile_path[0], file_name=file_name)
    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=vae_approx_path, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=fooocus_expansion_path,
        file_name='pytorch_model.bin'
    )
