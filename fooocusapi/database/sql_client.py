from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from fooocusapi.database.sql_model import GenerateRecord

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
    def __init__(self, backend: str, username: str, password: str,
                 host: str, port: int, database: str, db_path: str):
        if backend == 'mysql':
            self.engine = create_engine(
                f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")
        elif backend =='sqlite':
            self.engine = create_engine(
                f"sqlite:///{db_path}")
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
