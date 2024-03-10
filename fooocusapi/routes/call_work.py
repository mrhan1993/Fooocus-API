import asyncio
import json
import copy

from fastapi import Response
from fooocusapi.models.common.base import CommonRequest as Text2ImgRequest
from fooocusapi.models.common.response import TaskResponse
from fooocusapi.tasks.task_queue import task_queue
from fooocusapi.tasks.task import TaskObj
from fooocusapi.utils.img_utils import base64_to_bytesimg


async def call_worker(req: Text2ImgRequest,
                      accept: str,) -> Response | TaskResponse:
    """
    Call worker to generate image
    Args:
        req: Text2ImgRequest, or other object inherit CommonRequest
        accept: accept header
    returns: Response or TaskResponse
    """
    if accept == "image/png":
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    task = TaskObj(req_params=req)
    await task_queue.add_task(task=task)

    if streaming_output:
        while True:
            await asyncio.sleep(1)
            if task.task_status == 'success':
                print(task)
                try:
                    image_bytes = base64_to_bytesimg(task.task_result[0].base64)
                    return Response(
                        image_bytes,
                        media_type="image/png",
                    )
                except IndexError:
                    return Response(status_code=500, content="Internal Server Error")
            if task.task_status in ['failed', 'canceled']:
                return task.to_dict()

    if not req.async_process:
        while True:
            print("waiting for sync task")
            await asyncio.sleep(1)
            print(task.task_status)
            if task.task_status is not None:
                return task.to_dict()
    return task.to_dict()
