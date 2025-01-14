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
import json
from datetime import datetime
from fooocusapi.utils.logger import logger


output_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..', 'outputs', 'files'))
os.makedirs(output_dir, exist_ok=True)

STATIC_SERVER_BASE = 'http://127.0.0.1:3002/files/'
API_KEY = os.environ.get('API_KEY')
API_SECRET = os.environ.get('API_SECRET')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY")

client = Minio("minio-ew8sowww4gsogokg8okw00sk.154.26.133.70.sslip.io",
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
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
    current_time = datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")
    bucket_name = "otti-image-generation"

    # Check if bucket exists
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # Determine file path
    filename = os.path.join(date_string, image_name + '.' + extension)
    file_path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        if isinstance(img, str):
            # Move file to output directory
            shutil.move(img, file_path)
            # Upload to Minio
            client.fput_object(bucket_name, filename, file_path)
        else:
            # Convert np.ndarray to bytes
            image_data = narray_to_bytes(img)
            # Save image data to file
            with open(file_path, 'wb') as f:
                f.write(image_data)
            # Upload to Minio
            client.put_object(bucket_name, filename, BytesIO(image_data), length=len(image_data))

        # Upload to Pinata
        pinita_hash = upload_public_file_to_pinata(file_path, filename)

        print(
            file_path, "successfully uploaded as object",
            filename, "to bucket", bucket_name,
        )

         # Delete the local file after upload
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Local file {file_path} deleted")
        
        return pinita_hash
    except Exception as e:
        print("Error:", e)
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


def output_file_to_base64img(url: str) -> str | None:
    """
    Convert an image from a Pinata (IPFS) URL to a Base64 string.

    Args:
        url: The URL of the image on Pinata (IPFS).
    Returns:
        A Base64 string representation of the image or None if there's an error.
    """
    if not url:
        return None

    try:
        # Fetch the image from Pinata (IPFS)
        response = requests.get(f"https://lime-absent-tarsier-87.mypinata.cloud/ipfs/{url}" , stream=True)
        if response.status_code != 200:
            return None

        # Read the image content
        img = Image.open(BytesIO(response.content))
        
        # Determine the file format and convert to Base64
        ext = img.format.lower()  # Extract the format (e.g., PNG, JPEG)
        if ext not in ['png', 'jpg', 'jpeg', 'webp']:
            ext = 'png'  # Default to PNG if unsupported format
        
        # Save the image to a buffer
        output_buffer = BytesIO()
        img.save(output_buffer, format=ext.upper())
        byte_data = output_buffer.getvalue()
        
        # Encode the image to Base64
        base64_str = base64.b64encode(byte_data).decode('utf-8')
        return f"data:image/{ext};base64," + base64_str

    except Exception as e:
        print(f"Error: {e}")
        return None


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
    return "https://lime-absent-tarsier-87.mypinata.cloud/ipfs/" + filename

        
def upload_public_file_to_pinata(file_path, filename):
    # Endpoint untuk mengunggah file ke Pinata
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"

    # Membuka file yang ingin diunggah
    with open(file_path, "rb") as file:
        unique_name = f"Otti_{filename}"
        
        files = {
            "file": file,
            "pinataMetadata": (None, json.dumps({"name": unique_name})),
            "pinataOptions": (None, json.dumps({"cidVersion": 1}))
        }


        # Header otentikasi
        headers = {
            "Authorization": f"Bearer {JWT_SECRET_KEY}"
        }

        # Mengirim permintaan POST ke Pinata
        response = requests.post(url, files=files, headers=headers)

        # Menangani respons
        if response.status_code == 200:
            data = response.json()
            ipfs_hash = data["IpfsHash"]
            public_url = f"https://lime-absent-tarsier-87.mypinata.cloud/ipfs/{ipfs_hash}"
            return ipfs_hash
        else:
            print("Gagal mengunggah file!")
            print(response.json())
            return None


def narray_to_bytes(narray: np.ndarray) -> bytes:
    img = Image.fromarray(narray)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    return output_buffer.getvalue()
