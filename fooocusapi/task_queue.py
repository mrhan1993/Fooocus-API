from enum import Enum
import time
from typing import List


class TaskType(str, Enum):
    text2img = 'text2img'


class QueueTask(object):
    is_finished: bool = False
    start_millis: int = 0
    finish_millis: int = 0
    finish_with_error: bool = False
    task_result: any = None

    def __init__(self, seq: int, type: TaskType, req_param: dict, in_queue_millis: int):
        self.seq = seq
        self.type = type
        self.req_param = req_param
        self.in_queue_millis = in_queue_millis


class TaskQueue(object):
    queue: List[QueueTask] = []
    history: List[QueueTask] = []
    last_seq = 0

    def __init__(self, queue_size: int = 3):
        self.queue_size = queue_size

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
        self.last_seq += task.seq
        return task.seq

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

    def finish_task(self, seq: int, task_result: any, finish_with_error: bool):
        task = self.get_task(seq)
        if task is not None:
            task.is_finished = True
            task.finish_millis = int(round(time.time() * 1000))
            task.finish_with_error = finish_with_error
            task.task_result = task_result

            # Move task to history
            self.queue.remove(task)
            self.history.append(task)
