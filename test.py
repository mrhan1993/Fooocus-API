"""test code"""
import asyncio
import uuid
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


class QueueTask:
    """任务对象"""
    def __init__(self):
        self.task_id = str(uuid.uuid4())
        self.status = "pending"
        self.progress = 0

    async def run(self):
        """任务执行逻辑"""
        self.status = "running"
        for i in range(101):
            await asyncio.sleep(1)  # 模拟任务执行
            print(f"Task {self.task_id} is running, progress: {i}%")
            self.progress = i
        self.status = "completed"


class TaskQueue:
    """任务队列"""
    def __init__(self, queue_size: int):
        self.queue_size = queue_size
        self.queue = asyncio.Queue(queue_size)
        self.history = []

    async def add_task(self, task: QueueTask):
        """添加任务到队列"""
        if self.queue.full():
            print("队列已满，无法添加新任务")
            return
        await self.queue.put(task)

    async def process_tasks(self):
        """处理队列中的任务"""
        while True:
            if self.queue.empty():
                await asyncio.sleep(1)
                print("队列已空")
                continue
            else:
                task = await self.queue.get()
                await task.run()
                self.history.append(task)

    async def get_task_by_id(self, task_id: str) -> Optional[QueueTask]:
        """根据任务ID获取任务对象"""
        for task in list(self.queue._queue):  # pylint: disable=protected-access
            if task.task_id == task_id:
                return task
        # Check the history
        for task in self.history:
            if task.task_id == task_id:
                return task
        return None

    async def list_all_tasks(self):
        """获取所有任务列表"""
        tasks = list(self.queue._queue) + self.history  # pylint: disable=protected-access
        return tasks

task_queue = TaskQueue(100)


@asynccontextmanager
async def lifespan(app: FastAPI): # pylint: disable=unused-argument, redefined-outer-name
    """lifespan"""
    asyncio.create_task(task_queue.process_tasks())
    yield

app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(path='/v1/add_task')
async def add_task():
    """添加任务"""
    task = QueueTask()
    await task_queue.add_task(task)
    return {"status": "ok", "task_id": task.task_id}


@app.get(path='/v1/task_list')
async def task_list():
    """获取任务列表"""
    tasks = [{"task_id": task.task_id, "status": task.status, "progress": task.progress} for task in task_queue.history]
    return {"status": "ok", "tasks": tasks}


if __name__ == "__main__":
    uvicorn.run(app="test:app", host="0.0.0.0", port=8000, log_level="info")
