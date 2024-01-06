import os


db_conf = {
    "backend": "sqlite", # mysql, sqlite, redis
    "host": "localhost",
    "port": "3306",
    "user": "root",
    "password": "00000000",
    "database": "fooocus",
    "dbpath": os.path.join(os.path.dirname(__file__), "database.db").replace("\\", "/"),
}

redis_conf = {
    "host": "localhost",
    "port": "6379",
    "password": "foobared",
    "db": 0,
}

adv_params_keys = [
    "disable_preview",
    "adm_scaler_positive",
    "adm_scaler_negative",
    "adm_scaler_end",
    "adaptive_cfg",
    "sampler_name",
    "scheduler_name",
    "generate_image_grid",
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
    "inpaint_mask_upload_checkbox",
    "invert_mask_checkbox",
    "inpaint_erode_or_dilate"
]
