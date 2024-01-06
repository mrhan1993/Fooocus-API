import time
from fooocusapi.database.config import db_conf, adv_params_keys


if db_conf["backend"].lower() == "mysql" or db_conf["backend"].lower() == "sqlite":
    from fooocusapi.database.sql_client import MysqlSQLAlchemy
    db = MysqlSQLAlchemy(
        backend=db_conf["backend"],
        username=db_conf["user"],
        password=db_conf["password"],
        host=db_conf["host"],
        port=db_conf["port"],
        database=db_conf["database"],
        db_path=db_conf["dbpath"]
    )
    
elif db_conf["backend"].lower() == "redis":
    from fooocusapi.database.config import redis_conf
    from fooocusapi.database.redis_client import RedisClient
    db = RedisClient(
        host=redis_conf["host"],
        port=redis_conf["port"],
        password=redis_conf["password"],
        db=redis_conf["db"]
    )

def req_to_dict(req: dict) -> dict:
    req["loras"] = [{"model_name": lora[0], "weight": lora[1]} for lora in req["loras"]]
    req["advanced_params"] = dict(zip(adv_params_keys, req["advanced_params"]))
    req["image_prompts"] = [{
        "cn_img": "",
        "cn_stop": image[1],
        "cn_weight": image[2],
        "cn_type": image[3]
    } for image in req["image_prompts"]]
    del req["inpaint_input_image"]
    del req["uov_input_image"]
    return req

def add_history(params: dict, task_type: str, task_id: str, result_url: str, finish_reason: str) -> None:
    params = req_to_dict(params["params"])
    params["date_time"] = int(time.time())
    params["task_type"] = task_type
    params["task_id"] = task_id
    params["result_url"] = result_url
    params["finish_reason"] = finish_reason

    db.store_history(params)


def query_history(task_id: str=None, page: int=0, page_size: int=20, order_by: str="date_time"):
    return db.get_history(task_id=task_id, page=page, page_size=page_size, order_by=order_by)
