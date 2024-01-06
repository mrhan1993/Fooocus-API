from typing import Optional
from sqlalchemy import Integer, Float,VARCHAR, Boolean, JSON, Text, create_engine
from sqlalchemy.orm import declarative_base, Session, Mapped, mapped_column
from fooocusapi.database.config import db_conf


Base = declarative_base()

class GenerateRecord(Base):
    __tablename__ = 'generate_record'

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

if db_conf["backend"] == "mysql":
    engine = create_engine(f"mysql+pymysql://{db_conf['user']}:{db_conf['password']}@{db_conf['host']}:{db_conf['port']}/{db_conf['database']}")
if db_conf["backend"] == "sqlite":
    print(db_conf["dbpath"])
    engine = create_engine(f"sqlite:///{db_conf['dbpath']}")

session = Session(engine)
Base.metadata.create_all(engine, checkfirst=True)
session.close()
