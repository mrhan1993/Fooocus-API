"""Define task queue object"""
import asyncio
from typing import Optional

from fooocusapi.tasks.task import TaskObj


class TaskQueue:
    """任务队列"""
    def __init__(self, queue_size: int):
        self.queue_size = queue_size
        self.queue = asyncio.Queue(queue_size)
        self.current = None
        self.history = []
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(1)

    async def add_task(self, task: TaskObj):
        """添加任务到队列"""
        if self.queue.full():
            print("队列已满，无法添加新任务")
            return
        await self.queue.put(task)
        return task.to_dict()

    async def process_tasks(self):
        """处理队列中的任务"""
        while True:
            if self.queue.empty():
                await asyncio.sleep(1)
                print("队列已空")
                continue
            if self.current is not None:
                if self.current.task_status is not None:
                    self.current = None
                    continue
                await asyncio.sleep(1)
                print(f"{self.current.task_id} running...")
                continue
            async with self.semaphore:
                self.current = await self.queue.get()
                self.current.update('status', 'running')
                await asyncio.create_task(self.current.run())
                self.history.append(self.current)


task_queue = TaskQueue(100)
