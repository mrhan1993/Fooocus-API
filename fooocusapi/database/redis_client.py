import redis
import json
from datetime import datetime


class RedisClient:
    def __init__(self, host: str, port: str, password: str, db: int=15,) -> None:
        self.r = redis.Redis(host=host, port=port, password=password, db=db, decode_responses=True)

    @staticmethod
    def process_dict(data: dict) -> dict:
        task_info = {
            "task_id": data['task_id'],
            "task_type": data['task_type'],
            "result_url": data['result_url'],
            "finish_reason": data['finish_reason'],
            "date_time": datetime.fromtimestamp(data['date_time']).strftime("%Y-%m-%d %H:%M:%S"),
        }
        del data['task_id']
        del data['task_type']
        del data['result_url']
        del data['finish_reason']
        del data['date_time']
        return {"params": data, "task_info": task_info}

    def store_history(self, req: dict) -> None:
        task_id = req['task_id']
        data = self.process_dict(req)
        self.r.set(task_id, json.dumps(data))

    def get_history(self, task_id: str=None, page: int=0, page_size: int=20, order_by: str="date_time") -> list:
        """
        Get task history
        """
        if task_id is not None:
            data = self.r.get(task_id)
            if data == {}:
                return [{}]
            else:
                return [json.loads(data)]
        else:
            keys = self.r.keys("*")[page*page_size:(page+1)*page_size]
            if len(keys) > 0:
                result = self.r.mget(keys)
                return [json.loads(data) for data in result]
            else:
                return [{}]
