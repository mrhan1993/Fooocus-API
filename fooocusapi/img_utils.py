import base64
import requests
import numpy as np

from io import BytesIO
from fastapi import UploadFile
from PIL import Image


def narray_to_base64img(narray: np.ndarray) -> str:
    if narray is None:
        return None

    img = Image.fromarray(narray)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def narray_to_bytesimg(narray) -> bytes:
    if narray is None:
        return None

    img = Image.fromarray(narray)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    return byte_data


def read_input_image(input_image: UploadFile | None) -> np.ndarray | None:
    if input_image is None:
        return None
    input_image_bytes = input_image.file.read()
    pil_image = Image.open(BytesIO(input_image_bytes))
    image = np.array(pil_image)
    return image

def base64_to_stream(image: str) -> UploadFile | None:
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
    if url == '':
        return None
    try:
        response = requests.get(url, timeout=10)
        binary_image = response.content
    except:
        return None
    try:
        buffer = BytesIO(binary_image)
        Image.open(buffer)
    except:
        return None
    byte_stream = BytesIO()
    byte_stream.write(binary_image)
    byte_stream.seek(0)
    return UploadFile(file=byte_stream)
