import time

conf = {
    "backend": "mysql",
    "host": "192.168.63.246",
    "port": "3306",
    "user": "root",
    "password": "root",
    "database": "fooocus"
}

adv_params_keys = [
    "disable_preview",
    "adm_scaler_positive",
    "adm_scaler_negative",
    "adm_scaler_end",
    "adaptive_cfg",
    "sampler_name",
    "scheduler_name",
    "AP1",
    "overwrite_step",
    "overwrite_switch",
    "overwrite_width",
    "overwrite_height",
    "overwrite_vary_strength",
    "overwrite_upscale_strength",
    "mixing_image_prompt_and_vary_upscale",
    "mixing_image_prompt_and_inpaint",
    "debugging_cn_preprocessor",
    "skipping_cn_preprocessor",
    "controlnet_softness",
    "canny_low_threshold",
    "canny_high_threshold",
    "refiner_swap_method",
    "freeu_enabled",
    "freeu_b1",
    "freeu_b2",
    "freeu_s1",
    "freeu_s2",
    "debugging_inpaint_preprocessor",
    "inpaint_disable_initial_latent",
    "inpaint_engine",
    "inpaint_strength",
    "inpaint_respective_field",
    "AP2",
    "invert_mask_checkbox",
    "inpaint_erode_or_dilate"
]

def req_to_dict(req: dict) -> dict:
    req["loras"] = [{"model_name": lora[0], "weight": lora[1]} for lora in req["loras"]]
    req["advanced_params"] = zip(adv_params_keys, req["advanced_params"])
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
    params = req_to_dict(params)
    params["date_time"] = int(time.time())
    params["task_type"] = task_type
    params["task_id"] = task_id
    params["result_url"] = result_url
    params["finish_reason"] = finish_reason
    if conf["backend"].lower() == "mysql":
        from fooocusapi.database.sql_client import MysqlSQLAlchemy

        db = MysqlSQLAlchemy(
            username=conf["user"],
            password=conf["password"],
            host=conf["host"],
            port=conf["port"],
            database=conf["database"]
        )

        db.store_history(record=params)
