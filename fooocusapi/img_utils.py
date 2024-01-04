import base64
import numpy as np

from io import BytesIO
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
    if image.startswith('data:image'):
        image = image.split(sep=',', maxsplit=1)[1]
    image_bytes = base64.b64decode(image)
    byte_stream = BytesIO()
    byte_stream.write(image_bytes)
    byte_stream.seek(0)
    return UploadFile(file=byte_stream)
