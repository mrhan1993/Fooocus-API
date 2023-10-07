import base64
import inspect
import io
from io import BytesIO
from typing import Annotated

import numpy as np
from fastapi import Form, UploadFile
from PIL import Image


def narray_to_base64img(narray) -> str:
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


def read_input_image(input_image: UploadFile):
    input_image_bytes = input_image.file.read()
    pil_image = Image.open(io.BytesIO(input_image_bytes))
    return np.array(pil_image)


def as_form(cls):
    new_params = [
        inspect.Parameter(
            field_name,
            inspect.Parameter.POSITIONAL_ONLY,
            default=model_field.default,
            annotation=Annotated[model_field.annotation,
                                 *model_field.metadata, Form()],
        )
        for field_name, model_field in cls.model_fields.items()
    ]

    cls.__signature__ = cls.__signature__.replace(parameters=new_params)

    return cls


class QueueReachLimitException(Exception):
    pass
