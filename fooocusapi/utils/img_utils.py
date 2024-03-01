"""Image process utils. Used to verify, convert and store Images."""
# pylint: disable=broad-exception-caught


import base64
from io import BytesIO

import requests
import numpy as np

from fastapi import UploadFile
from PIL import Image


def narray_to_base64img(narray: np.ndarray) -> str:
    """
    Convert numpy array to base64 image string.
    Args:
        narray: numpy array
    Returns:
        base64 image string
    """
    if narray is None:
        return None

    img = Image.fromarray(narray)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def narray_to_bytesimg(narray: np.ndarray) -> bytes:
    """
    Convert numpy array to bytes image.
    Args:
        narray: numpy array
    Returns:
        bytes image
    """
    if narray is None:
        return None

    img = Image.fromarray(narray)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    return byte_data


def read_input_image(input_image: UploadFile | None) -> np.ndarray | None:
    """
    Read input image from UploadFile.
    Args:
        input_image: UploadFile
    Returns:
        numpy array of image
    """
    if input_image is None:
        return None
    input_image_bytes = input_image.file.read()
    pil_image = Image.open(BytesIO(input_image_bytes))
    image = np.array(pil_image)
    return image


def base64_to_stream(image: str) -> UploadFile | None:
    """
    Convert base64 image string to UploadFile.
    Args:
        image: base64 image string
    Returns:
        UploadFile or None
    """
    if image == '':
        return None
    if image.startswith('http'):
        return get_check_image(url=image)
    if image.startswith('data:image'):
        image = image.split(sep=',', maxsplit=1)[1]
    image_bytes = base64.b64decode(image)
    byte_stream = BytesIO()
    byte_stream.write(image_bytes)
    byte_stream.seek(0)
    return UploadFile(file=byte_stream)

def get_check_image(url: str) -> UploadFile | None:
    """
    Get image from url and check if it's valid.
    Args:
        url: image url
    Returns:
        UploadFile or None
    """
    if url == '':
        return None
    try:
        response = requests.get(url, timeout=10)
        binary_image = response.content
    except Exception:
        return None
    try:
        buffer = BytesIO(binary_image)
        Image.open(buffer)
    except Exception:
        return None
    byte_stream = BytesIO()
    byte_stream.write(binary_image)
    byte_stream.seek(0)
    return UploadFile(file=byte_stream)


def bytes_image_to_io(binary_image: bytes) -> BytesIO | None:
    """
    Convert bytes image to BytesIO.
    Args:
        binary_image: bytes image
    Returns:
        BytesIO or None
    """
    try:
        buffer = BytesIO(binary_image)
        Image.open(buffer)
    except Exception:
        return None
    byte_stream = BytesIO()
    byte_stream.write(binary_image)
    byte_stream.seek(0)
    return byte_stream


def bytes_to_base64img(byte_data: bytes) -> str | None:
    """
    Convert bytes image to base64 image string.
    Args:
        byte_data: bytes image
    Returns:
        base64 image string or None
    """
    if byte_data is None:
        return None

    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str
