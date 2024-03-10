"""Define task object"""
import json
import copy
import concurrent.futures
import asyncio
import threading
from typing import List
import time
import uuid

from fooocusapi.hooks.pre_task import pre_task
from fooocusapi.hooks.post_task import post_task

from fooocusapi.models.common.task import (
    TaskType,
    GeneratedImageResult
)
from fooocusapi.utils.api_utils import req_to_params, get_task_type
from fooocusapi.workers.worker import process_generate, process_stop


class TaskObj:
    """Task object"""
    def __init__(self, req_params: object):
        self.accept: str = "application/json"
        self.task_id: str = str(uuid.uuid4())
        self.task_type: TaskType | None = None
        self.req_param: object = req_params
        self.in_queue_millis: int = int(time.time()*1000)
        self.start_millis: int = 0
        self.finish_millis: int = 0
        self.status: str = "pending"  # 任务状态，可以是 "pending", "running", "completed"
        self.task_status: str | None = None  # "success", "failed", "canceled"
        self.progress: int = 0  # 任务进度，0到100
        self.task_step_preview: str | None = None
        self.webhook_url: str | None = None
        self.task_result: List[GeneratedImageResult] = []

    def update(self, attribute: str, value):
        """
        Update task obj
        Args:
            attribute: attribute name
            value: value
        """
        setattr(self, attribute, value)
        return getattr(self, attribute)

    async def pre_task(self):
        """pre task"""
        setattr(self, "req_param", pre_task(self.req_param))

    async def run(self):
        """start running task"""
        self.update("status", "running")
        self.update("task_type", get_task_type(self.req_param))
        params = req_to_params(self.req_param)
        sem = threading.Semaphore()
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(executor, process_generate, self, params)
                await future
        finally:
            sem.release()
        # await process_generate(self, params)
        self.update("status", "completed")

    async def stop(self):
        """stop running task"""
        self.update("status", "canceled")
        process_stop()
        return self.to_dict()

    async def post_task(self):
        """post task"""
        post_task(self.to_dict())

    def to_dict(self):
        """
        Copy object to a temp variable,
            Converts all values of type object to dictionary
        :return: Dict for this object
        """
        obj_dict = copy.deepcopy(self.__dict__)

        # Convert Enum obj to value
        try:
            obj_dict['task_type'] = obj_dict['task_type'].value
        except:
            pass

        # Convert loras tuple to dict
        loras = []
        for lora in obj_dict['req_param'].loras:
            try:
                lora = {
                    'model_name': lora[0],
                    'weight': lora[1]
                }
            except Exception:
                lora = {
                    'model_name': lora.model_name,
                    'weight': lora.weight
                }
            loras.append(lora)
        obj_dict['req_param'].loras = loras

        # if image prompts exist, convert to dict, now is set to None
        try:
            obj_dict['req_param'].image_prompts = None
        except (AttributeError, ValueError):
            pass

        obj_dict['req_param'] = json.loads(obj_dict['req_param'].model_dump_json())
        if len(obj_dict['task_result']) > 0:
            task_result = []
            for result in obj_dict['task_result']:
                result = json.loads(result.model_dump_json())
                task_result.append(result)
            obj_dict['task_result'] = task_result
        return obj_dict
    
    def __str__(self):
        return f"QueueTask(task_id={self.task_id}, task_type={self.task_type}, \
                in_queue_millis={self.in_queue_millis}, start_millis={self.start_millis}, \
                finish_millis={self.finish_millis}, status={self.status}, \
                progress={self.progress}, task_status={self.task_status}, \
                task_step_preview={self.task_step_preview}, task_result={self.task_result})"
