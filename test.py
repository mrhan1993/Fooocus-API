"""test code"""
import os
import sys

# add fooocus api and fooocus to system path
script_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(script_path, 'repositories/Fooocus')

sys.path.append(script_path)
sys.path.append(module_path)

from contextlib import asynccontextmanager
import asyncio
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fooocusapi.tasks.task import TaskObj
from fooocusapi.tasks.task_queue import TaskQueue

from fooocusapi.models.common.base import CommonRequest as Text2ImgRequest


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


@app.post(path='/v1/generation/test')
async def add_task(request: Text2ImgRequest):
    """添加任务"""
    task = TaskObj(req_params=request, webhook_url=None)
    await task_queue.add_task(task)
    return task.to_dict()


@app.get(path='/v1/task_list')
async def task_list():
    """获取任务列表"""
    tasks = [{"task_id": task.task_id, "status": task.status,
              "progress": task.progress} for task in task_queue.history]
    return {"status": "ok", "tasks": tasks}


if __name__ == "__main__":
    uvicorn.run(app="test:app", host="0.0.0.0", port=8888, log_level="info")
