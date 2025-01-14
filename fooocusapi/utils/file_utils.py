# -*- coding: utf-8 -*-

""" File utils

Use for managing generated files

@file: file_utils.py
@author: Konie
@update: 2024-03-22
"""
import base64
import datetime
import shutil
from io import BytesIO
import os
from pathlib import Path
import numpy as np
from PIL import Image
from minio import Minio
import requests


from fooocusapi.utils.logger import logger


output_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..', 'outputs', 'files'))
os.makedirs(output_dir, exist_ok=True)

STATIC_SERVER_BASE = 'http://127.0.0.1:3002/files/'
API_KEY = "6c8b6b3e68c3a52eff8d"
API_SECRET = "71dd146ef4b6fc8b642093d4b187225db9a513eca8a4f143c151e7026f9c50e9"

client = Minio("minio-ew8sowww4gsogokg8okw00sk.154.26.133.70.sslip.io",
    access_key="s3bTdpCkT3IqVGlB",
    secret_key="FyJXS9xiDAMbBBFQjryLuQrDn14ROLBE",
)


def save_output_file(
        img: np.ndarray | str,
        image_name: str = '',
        extension: str = 'png') -> str:
    """
    Save np image to file
    Args:
        img: np.ndarray image to save
        image_name: str of image name
        extension: str of image extension
    Returns:
        str of file name
    """
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")
    bucket_name =  "otti-image-generation"
    image_data = narray_to_bytes(img) 
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

        
        
    filename = os.path.join(date_string, image_name + '.' + extension)
    file_path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        if isinstance(img, str):
            # shutil.move(img, file_path)
         client.fput_object(
            bucket_name, file_path, img,
        )
        upload_image_data_to_pinata(image_data)
        print(
            file_path, "successfully uploaded as object",
            filename, "to bucket", bucket_name,
        )
        return Path(file_path).as_posix()
    except Exception:
        raise Exception


def delete_output_file(filename: str):
    """
    Delete files specified in the output directory
    Args:
        filename: str of file name
    """
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        logger.std_warn(f'[Fooocus API] {filename} not exists or is not a file')
    try:
        os.remove(file_path)
        logger.std_info(f'[Fooocus API] Delete output file: {filename}')
        return True
    except OSError:
        logger.std_error(f'[Fooocus API] Delete output file failed: {filename}')
        return False


def output_file_to_base64img(filename: str | None) -> str | None:
    """
    Convert an image file to a base64 string.
    Args:
        filename: str of file name
    return: str of base64 string
    """
    if filename is None:
        return None
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return None

    ext = filename.split('.')[-1]
    if ext.lower() not in ['png', 'jpg', 'webp', 'jpeg']:
        ext = 'png'
    img = Image.open(file_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format=ext.upper())
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return f"data:image/{ext};base64," + base64_str


def output_file_to_bytesimg(filename: str | None) -> bytes | None:
    """
    Convert an image file to a bytes string.
    Args:
        filename: str of file name
    return: bytes of image data
    """
    if filename is None:
        return None
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return None

    img = Image.open(file_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    return byte_data


def get_file_serve_url(filename: str | None) -> str | None:
    """
    Get the static serve url of an image file.
    Args:
        filename: str of file name
    return: str of static serve url
    """
    if filename is None:
        return None
    return "https://console-ew8sowww4gsogokg8okw00sk.154.26.133.70.sslip.io/otti-image-generation/files" + '/'.join(filename.split('/')[-2:])


def upload_image_data_to_pinata(image_data: bytes):
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"

    # Header otentikasi
    headers = {
        "pinata_api_key": API_KEY,
        "pinata_secret_api_key": API_SECRET,
    }

    # Mengirim permintaan
    files = {"file": ("image.png", image_data, "image/png")}
    response = requests.post(url, files=files, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print("File berhasil diunggah:")
        print("IPFS Hash:", data["IpfsHash"])
        print("Size:", data["PinSize"], "bytes")
        return data
    else:
        print("Gagal mengunggah file:")
        print(response.json())
        return None
        
def upload_public_file_to_pinata(file_path):
    # Endpoint untuk mengunggah file ke Pinata
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"

    # Membuka file yang ingin diunggah
    with open(file_path, "rb") as file:
        files = {"file": file}

        # Header otentikasi
        headers = {
            "pinata_api_key": API_KEY,
            "pinata_secret_api_key": API_SECRET,
        }

        # Mengirim permintaan POST ke Pinata
        response = requests.post(url, files=files, headers=headers)

        # Menangani respons
        if response.status_code == 200:
            data = response.json()
            ipfs_hash = data["IpfsHash"]
            public_url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
            print("File berhasil diunggah!")
            print("IPFS Hash:", ipfs_hash)
            print("URL Publik:", public_url)
            return public_url
        else:
            print("Gagal mengunggah file!")
            print(response.json())
            return None

def get_private_file_from_pinata(ipfs_hash):
    gateway_url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"

    headers = {
        "pinata_api_key": API_KEY,
        "pinata_secret_api_key": API_SECRET,
    }

    response = requests.get(gateway_url, headers=headers)

    if response.status_code == 200:
        print("File berhasil diakses:")
        print(response.content)  # Atau simpan file ini ke disk jika diperlukan
    else:
        print("Gagal mengakses file:")
        print(response.json())

def narray_to_bytes(narray: np.ndarray) -> bytes:
    img = Image.fromarray(narray)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    return output_buffer.getvalue()
