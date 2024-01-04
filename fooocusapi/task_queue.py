from enum import Enum
import time
import numpy as np
import uuid
from typing import List, Tuple
import requests
from fooocusapi.file_utils import delete_output_file, get_file_serve_url

from fooocusapi.img_utils import narray_to_base64img
from fooocusapi.parameters import ImageGenerationResult, GenerationFinishReason


class TaskType(str, Enum):
    text_2_img = 'Text to Image'
    img_uov = 'Image Upscale or Variation'
    img_inpaint_outpaint = 'Image Inpaint or Outpaint'
    img_prompt = 'Image Prompt'
    not_found = 'Not Found'


class QueueTask(object):
    job_id: str
    is_finished: bool = False
    finish_progress: int = 0
    start_millis: int = 0
    finish_millis: int = 0
    finish_with_error: bool = False
    task_status: str | None = None
    task_step_preview: str | None = None
    task_result: List[ImageGenerationResult] = None
    error_message: str | None = None
    webhook_url: str | None = None  # attribute for individual webhook_url

    def __init__(self, job_id: str, type: TaskType, req_param: dict, in_queue_millis: int,
                 webhook_url: str | None = None):
        self.job_id = job_id
        self.type = type
        self.req_param = req_param
        self.in_queue_millis = in_queue_millis
        self.webhook_url = webhook_url

    def set_progress(self, progress: int, status: str | None):
        if progress > 100:
            progress = 100
        self.finish_progress = progress
        self.task_status = status

    def set_step_preview(self, task_step_preview: str | None):
        self.task_step_preview = task_step_preview

    def set_result(self, task_result: List[ImageGenerationResult], finish_with_error: bool, error_message: str | None = None):
        if not finish_with_error:
            self.finish_progress = 100
            self.task_status = 'Finished'
        self.task_result = task_result
        self.finish_with_error = finish_with_error
        self.error_message = error_message


class TaskQueue(object):
    queue: List[QueueTask] = []
    history: List[QueueTask] = []
    last_job_id = None
    webhook_url: str | None = None

    def __init__(self, queue_size: int, hisotry_size: int, webhook_url: str | None = None):
        self.queue_size = queue_size
        self.history_size = hisotry_size
        self.webhook_url = webhook_url

    def add_task(self, type: TaskType, req_param: dict, webhook_url: str | None = None) -> QueueTask | None:
        """
        Create and add task to queue
        :returns: The created task's job_id, or None if reach the queue size limit
        """
        if len(self.queue) >= self.queue_size:
            return None

        job_id = str(uuid.uuid4())
        task = QueueTask(job_id=job_id, type=type, req_param=req_param,
                         in_queue_millis=int(round(time.time() * 1000)),
                         webhook_url=webhook_url)
        self.queue.append(task)
        self.last_job_id = job_id
        return task

    def get_task(self, job_id: str, include_history: bool = False) -> QueueTask | None:
        for task in self.queue:
            if task.job_id == job_id:
                return task

        if include_history:
            for task in self.history:
                if task.job_id == job_id:
                    return task

        return None

    def is_task_ready_to_start(self, job_id: str) -> bool:
        task = self.get_task(job_id)
        if task is None:
            return False

        return self.queue[0].job_id == job_id

    def start_task(self, job_id: str):
        task = self.get_task(job_id)
        if task is not None:
            task.start_millis = int(round(time.time() * 1000))

    def finish_task(self, job_id: str):
        task = self.get_task(job_id)
        if task is not None:
            task.is_finished = True
            task.finish_millis = int(round(time.time() * 1000))

            # Use the task's webhook_url if available, else use the default
            webhook_url = task.webhook_url or self.webhook_url

            # Send webhook
            if task.is_finished and webhook_url:
                data = { "job_id": task.job_id, "job_result": [] }
                if isinstance(task.task_result, List):
                    for item in task.task_result:
                        data["job_result"].append({
                            "url": get_file_serve_url(item.im) if item.im else None,
                            "seed": item.seed if item.seed else "-1",
                        })
                try:
                    res = requests.post(webhook_url, json=data)
                    print(f'Call webhook response status: {res.status_code}')
                except Exception as e:
                    print('Call webhook error:', e)

            # Move task to history
            self.queue.remove(task)
            self.history.append(task)

            # Clean history
            if len(self.history) > self.history_size and self.history_size != 0:
                removed_task = self.history.pop(0)
                if isinstance(removed_task.task_result, List):
                    for item in removed_task.task_result:
                        if isinstance(item, ImageGenerationResult) and item.finish_reason == GenerationFinishReason.success and item.im is not None:
                            delete_output_file(item.im)
                print(f"Clean task history, remove task: {removed_task.job_id}")


class TaskOutputs:
    outputs = []

    def __init__(self, task: QueueTask):
        self.task = task

    def append(self, args: List[any]):
        self.outputs.append(args)
        if len(args) >= 2:
            if args[0] == 'preview' and isinstance(args[1], Tuple) and len(args[1]) >= 2:
                number = args[1][0]
                text = args[1][1]
                self.task.set_progress(number, text)
                if len(args[1]) >= 3 and isinstance(args[1][2], np.ndarray):
                    base64_preview_img = narray_to_base64img(args[1][2])
                    self.task.set_step_preview(base64_preview_img)