from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sql_model import GenerateRecord

def convert_to_dict_list(obj_list: list[object]) -> dict:
    dict_list = []
    for obj in obj_list:
        # 将对象属性转化为字典键值对
        dict_obj = {}
        for attr, value in vars(obj).items():
            if not callable(value) and not attr.startswith("__") and not attr.startswith("_"):
                dict_obj[attr] = value
        task_info = {
            "task_id": obj.task_id,
            "task_type": obj.task_type,
            "result_url": obj.result_url,
            "finish_reason": obj.finish_reason,
            "date_time": datetime.fromtimestamp(obj.date_time).strftime("%Y-%m-%d %H:%M:%S"),
        }
        del dict_obj['task_id']
        del dict_obj['task_type']
        del dict_obj['result_url']
        del dict_obj['finish_reason']
        del dict_obj['date_time']
        dict_list.append({"params": dict_obj, "task_info": task_info})
    return dict_list

class MysqlSQLAlchemy:
    def __init__(self, username: str, password: str, host: str, port: int, database: str):
        self.engine = create_engine(
            f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}",)
        self.session = Session(self.engine)

    def store_history(self, record: dict) -> None:
        """
        Store history to database
        :param record:
        :return:
        """
        self.session.add_all([GenerateRecord(**record)])
        self.session.commit()
    
    def get_history(self, task_id: str=None, page: int=0, page_size: int=20,
                    order_by: str='date_time') -> list:
        """
        Get history from database
        :param task_id:
        :return:
        """
        if task_id is not None:
            res = self.session.query(GenerateRecord).filter(GenerateRecord.task_id == task_id).all()
            if len(res) == 0:
                return []
            return convert_to_dict_list(res)
        
        res = self.session.query(GenerateRecord).order_by(getattr(GenerateRecord, order_by).desc()).offset(page * page_size).limit(page_size).all()
        if len(res) == 0:
            return []
        return convert_to_dict_list(res)


# upscale_params = {
#   "task_id": "ssssss",
#   "task_type": "upscale",
#   "prompt": "",
#   "negative_prompt": "",
#   "style_selections": [
#     "Fooocus V2",
#     "Fooocus Enhance",
#     "Fooocus Sharp"
#   ],
#   "performance_selection": "Speed",
#   "aspect_ratios_selection": "1152*896",
#   "image_number": 1,
#   "image_seed": -1,
#   "sharpness": 2,
#   "guidance_scale": 4,
#   "base_model_name": "juggernautXL_version6Rundiffusion.safetensors",
#   "refiner_model_name": "None",
#   "refiner_switch": 0.5,
#   "loras": [
#     {
#       "model_name": "sd_xl_offset_example-lora_1.0.safetensors",
#       "weight": 0.1
#     }
#   ],
#   "advanced_params": {
#     "disable_preview": False,
#     "adm_scaler_positive": 1.5,
#     "adm_scaler_negative": 0.8,
#     "adm_scaler_end": 0.3,
#     "refiner_swap_method": "joint",
#     "adaptive_cfg": 7,
#     "sampler_name": "dpmpp_2m_sde_gpu",
#     "scheduler_name": "karras",
#     "overwrite_step": -1,
#     "overwrite_switch": -1,
#     "overwrite_width": -1,
#     "overwrite_height": -1,
#     "overwrite_vary_strength": -1,
#     "overwrite_upscale_strength": -1,
#     "mixing_image_prompt_and_vary_upscale": False,
#     "mixing_image_prompt_and_inpaint": False,
#     "debugging_cn_preprocessor": False,
#     "skipping_cn_preprocessor": False,
#     "controlnet_softness": 0.25,
#     "canny_low_threshold": 64,
#     "canny_high_threshold": 128,
#     "freeu_enabled": False,
#     "freeu_b1": 1.01,
#     "freeu_b2": 1.02,
#     "freeu_s1": 0.99,
#     "freeu_s2": 0.95,
#     "debugging_inpaint_preprocessor": False,
#     "inpaint_disable_initial_latent": False,
#     "inpaint_engine": "v1",
#     "inpaint_strength": 1,
#     "inpaint_respective_field": 1
#   },
#   "require_base64": False,
#   "async_process": False,
#   "uov_method": "Upscale (2x)",
#   "input_image": ""
# }
# import json
# sql = MysqlSQLAlchemy()
# # print(sql.get_history(task_id='ssssss'))
# print(json.dumps(sql.get_history(page=0, page_size=20),indent=4, ensure_ascii=True))
# sql.store_history(record=upscale_params)
