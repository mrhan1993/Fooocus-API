"""
SQLite client for Fooocus API
"""
import os
import time
import platform
from datetime import datetime
from typing import Optional
import copy

from sqlalchemy import Integer, Float, VARCHAR, Boolean, JSON, Text, create_engine
from sqlalchemy.orm import declarative_base, Session, Mapped, mapped_column


Base = declarative_base()


if platform.system().lower() == "windows":
    default_sqlite_db_path = os.path.join(
        os.path.dirname(__file__), "../database.db"
    ).replace("\\", "/")
else:
    default_sqlite_db_path = os.path.join(os.path.dirname(__file__), "../database.db")

connection_uri = os.environ.get(
    "FOOOCUS_DB_CONF", f"sqlite:///{default_sqlite_db_path}"
)


class GenerateRecord(Base):
    """
    GenerateRecord

    __tablename__ = 'generate_record'
    """

    __tablename__ = "generate_record"

    task_id: Mapped[str] = mapped_column(VARCHAR(255), nullable=False, primary_key=True)
    task_type: Mapped[str] = mapped_column(Text, nullable=False)
    result_url: Mapped[str] = mapped_column(Text, nullable=True)
    finish_reason: Mapped[str] = mapped_column(Text, nullable=True)
    date_time: Mapped[int] = mapped_column(Integer, nullable=False)

    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    negative_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    style_selections: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    performance_selection: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    aspect_ratios_selection: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    base_model_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    refiner_model_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    refiner_switch: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    loras: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    image_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sharpness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    guidance_scale: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    advanced_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    input_image: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    input_mask: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    image_prompts: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    inpaint_additional_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    outpaint_selections: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    outpaint_distance_left: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    outpaint_distance_right: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    outpaint_distance_top: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    outpaint_distance_bottom: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    uov_method: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    upscale_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    webhook_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    require_base64: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    async_process: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    def __repr__(self) -> str:
        return f"GenerateRecord(task_id={self.task_id!r}, task_type={self.task_type!r}, \
                result_url={self.result_url!r}, finish_reason={self.finish_reason!r}, date_time={self.date_time!r}, \
                prompt={self.prompt!r}, negative_prompt={self.negative_prompt!r}, style_selections={self.style_selections!r}, performance_selection={self.performance_selection!r}, \
                aspect_ratios_selection={self.aspect_ratios_selection!r}, base_model_name={self.base_model_name!r}, \
                refiner_model_name={self.refiner_model_name!r}, refiner_switch={self.refiner_switch!r}, loras={self.loras!r}, \
                image_number={self.image_number!r}, image_seed={self.image_seed!r}, sharpness={self.sharpness!r}, \
                guidance_scale={self.guidance_scale!r}, advanced_params={self.advanced_params!r}, input_image={self.input_image!r}, \
                input_mask={self.input_mask!r}, image_prompts={self.image_prompts!r}, inpaint_additional_prompt={self.inpaint_additional_prompt!r}, \
                outpaint_selections={self.outpaint_selections!r}, outpaint_distance_left={self.outpaint_distance_left!r}, outpaint_distance_right={self.outpaint_distance_right!r}, \
                outpaint_distance_top={self.outpaint_distance_top!r}, outpaint_distance_bottom={self.outpaint_distance_bottom!r}, uov_method={self.uov_method!r}, \
                upscale_value={self.upscale_value!r}, webhook_url={self.webhook_url!r}, require_base64={self.require_base64!r}, \
                async_process={self.async_process!r})"


engine = create_engine(connection_uri)

session = Session(engine)
Base.metadata.create_all(engine, checkfirst=True)
session.close()


def convert_to_dict_list(obj_list: list[object]) -> list[dict]:
    """
    Convert a list of objects to a list of dictionaries.
    Args:
        obj_list:

    Returns:
        dict_list:
    """
    dict_list = []
    for obj in obj_list:
        # 将对象属性转化为字典键值对
        dict_obj = {}
        for attr, value in vars(obj).items():
            if (
                not callable(value)
                and not attr.startswith("__")
                and not attr.startswith("_")
            ):
                dict_obj[attr] = value
        task_info = {
            "task_id": obj.task_id,
            "task_type": obj.task_type,
            "result_url": obj.result_url,
            "finish_reason": obj.finish_reason,
            "date_time": datetime.fromtimestamp(obj.date_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        del dict_obj["task_id"]
        del dict_obj["task_type"]
        del dict_obj["result_url"]
        del dict_obj["finish_reason"]
        del dict_obj["date_time"]
        dict_list.append({"params": dict_obj, "task_info": task_info})
    return dict_list


class MySQLAlchemy:
    """
    MySQLAlchemy, a toolkit for managing SQLAlchemy connections and sessions.

    :param uri: SQLAlchemy connection URI
    """

    def __init__(self, uri: str):
        # 'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
        self.engine = create_engine(uri)
        self.session = Session(self.engine)

    def store_history(self, record: dict) -> None:
        """
        Store history to database
        :param record:
        :return:
        """
        self.session.add_all([GenerateRecord(**record)])
        self.session.commit()

    def get_history(
        self,
        task_id: str = None,
        page: int = 0,
        page_size: int = 20,
        order_by: str = "date_time",
    ) -> list:
        """
        Get history from database
        :param task_id:
        :param page:
        :param page_size:
        :param order_by:
        :return:
        """
        if task_id is not None:
            res = (
                self.session.query(GenerateRecord)
                .filter(GenerateRecord.task_id == task_id)
                .all()
            )
            if len(res) == 0:
                return []
            return convert_to_dict_list(res)

        res = (
            self.session.query(GenerateRecord)
            .order_by(getattr(GenerateRecord, order_by).desc())
            .offset(page * page_size)
            .limit(page_size)
            .all()
        )
        if len(res) == 0:
            return []
        return convert_to_dict_list(res)


db = MySQLAlchemy(uri=connection_uri)


def req_to_dict(req: dict) -> dict:
    """
    Convert request to dictionary
    Args:
        req:

    Returns:

    """
    req["loras"] = [{"model_name": lora[0], "weight": lora[1]} for lora in req["loras"]]
    # req["advanced_params"] = dict(zip(adv_params_keys, req["advanced_params"]))
    req["image_prompts"] = [
        {"cn_img": "", "cn_stop": image[1], "cn_weight": image[2], "cn_type": image[3]}
        for image in req["image_prompts"]
    ]
    del req["inpaint_input_image"]
    del req["uov_input_image"]
    return req


def add_history(
    params: dict, task_type: str, task_id: str, result_url: str, finish_reason: str
) -> None:
    """
    Store history to database
    Args:
        params:
        task_type:
        task_id:
        result_url:
        finish_reason:

    Returns:

    """
    adv = copy.deepcopy(params["advanced_params"])
    params["advanced_params"] = adv.__dict__
    params["date_time"] = int(time.time())
    params["task_type"] = task_type
    params["task_id"] = task_id
    params["result_url"] = result_url
    params["finish_reason"] = finish_reason

    del params["inpaint_input_image"]
    del params["uov_input_image"]
    del params["save_extension"]
    del params["save_meta"]
    del params["save_name"]
    del params["meta_scheme"]

    db.store_history(params)


def query_history(
        task_id: str = None,
        page: int = 0,
        page_size: int = 20,
        order_by: str = "date_time"
) -> list:
    """
    Query history from database
    Args:
        task_id:
        page:
        page_size:
        order_by:

    Returns:

    """
    return db.get_history(
        task_id=task_id, page=page, page_size=page_size, order_by=order_by
    )
