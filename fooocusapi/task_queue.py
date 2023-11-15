from enum import Enum
import time
import numpy as np
from typing import List, Tuple

from fooocusapi.img_utils import narray_to_base64img


class TaskType(str, Enum):
    text_2_img = 'Text to Image'
    img_uov = 'Image Upscale or Variation'
    img_inpaint_outpaint = 'Image Inpaint or Outpaint'
    img_prompt = 'Image Prompt'


class QueueTask(object):
    is_finished: bool = False
    finish_progess: int = 0
    start_millis: int = 0
    finish_millis: int = 0
    finish_with_error: bool = False
    task_status: str | None = None
    task_step_preview: str | None = None
    task_result: any = None
    error_message: str | None = None

    def __init__(self, seq: int, type: TaskType, req_param: dict, in_queue_millis: int):
        self.seq = seq
        self.type = type
        self.req_param = req_param
        self.in_queue_millis = in_queue_millis

    def set_progress(self, progress: int, status: str | None):
        if progress > 100:
            progress = 100
        self.finish_progess = progress
        self.task_status = status

    def set_step_preview(self, task_step_preview: str | None):
        self.task_step_preview = task_step_preview

    def set_result(self, task_result: any, finish_with_error: bool, error_message: str | None = None):
        if not finish_with_error:
            self.finish_progess = 100
            self.task_status = 'Finished'
        self.task_result = task_result
        self.finish_with_error = finish_with_error
        self.error_message = error_message


class TaskQueue(object):
    queue: List[QueueTask] = []
    history: List[QueueTask] = []
    last_seq = 0

    def __init__(self, queue_size: int, hisotry_size: int):
        self.queue_size = queue_size
        self.history_size = hisotry_size

    def add_task(self, type: TaskType, req_param: dict) -> QueueTask | None:
        """
        Create and add task to queue
        :returns: The created task's seq, or None if reach the queue size limit
        """
        if len(self.queue) >= self.queue_size:
            return None

        task = QueueTask(seq=self.last_seq+1, type=type, req_param=req_param,
                         in_queue_millis=int(round(time.time() * 1000)))
        self.queue.append(task)
        self.last_seq = task.seq
        return task

    def get_task(self, seq: int, include_history: bool = False) -> QueueTask | None:
        for task in self.queue:
            if task.seq == seq:
                return task

        if include_history:
            for task in self.history:
                if task.seq == seq:
                    return task

        return None

    def is_task_ready_to_start(self, seq: int) -> bool:
        task = self.get_task(seq)
        if task is None:
            return False

        return self.queue[0].seq == seq

    def start_task(self, seq: int):
        task = self.get_task(seq)
        if task is not None:
            task.start_millis = int(round(time.time() * 1000))

    def finish_task(self, seq: int):
        task = self.get_task(seq)
        if task is not None:
            task.is_finished = True
            task.finish_millis = int(round(time.time() * 1000))

            # Move task to history
            self.queue.remove(task)
            self.history.append(task)

            # Clean history
            if len(self.history) > self.history_size:
                removed_task = self.history.pop(0)
                print(f"Clean task history, remove task: {removed_task.seq}")


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