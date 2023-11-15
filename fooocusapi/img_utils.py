import base64
from io import BytesIO
import numpy as np
from fastapi import Response, UploadFile
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


def read_input_image(input_image: UploadFile) -> np.ndarray:
    input_image_bytes = input_image.file.read()
    pil_image = Image.open(BytesIO(input_image_bytes))
    image = np.array(pil_image)
    return image