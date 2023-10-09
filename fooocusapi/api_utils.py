import base64
import inspect
import io
from io import BytesIO
from typing import Annotated, List

import numpy as np
from fastapi import Form, Response, UploadFile
from PIL import Image
from fooocusapi.models import GeneratedImage, GeneratedImageBase64, GenerationFinishReason

from modules.util import HWC3


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
    pil_image = Image.open(io.BytesIO(input_image_bytes))
    image = np.array(pil_image)
    return image


def generation_output(results: List[GeneratedImage], streaming_output: bool) -> Response | List[GeneratedImageBase64]:
    if streaming_output:
        if len(results) == 0 or results[0].finish_reason is not GenerationFinishReason.success:
            return Response(status_code=500)
        bytes = narray_to_bytesimg(results[0].im)
        return Response(bytes, media_type='image/png')
    else:
        results = [GeneratedImageBase64(base64=narray_to_base64img(
            item.im), seed=item.seed, finish_reason=item.finish_reason) for item in results]
        return results


class QueueReachLimitException(Exception):
    pass
