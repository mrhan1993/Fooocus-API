import copy
from enum import Enum
import random
from typing import List, Annotated
from fastapi import FastAPI, Header, Response
from pydantic import BaseModel, Field
import uvicorn
from fooocusapi.api_utils import narray_to_bytesimg
from modules.expansion import safe_str
from modules.sdxl_styles import apply_style, fooocus_expansion, aspect_ratios

app = FastAPI()

class Lora(BaseModel):
    model_name: str
    weight: float

class PerfomanceSelection(str, Enum):
    speed = 'Speed'
    quality = 'Quality'

class FooocusStyle(str, Enum):
    fooocus_expansion = 'Fooocus V2'
    default = 'Default (Slightly Cinematic)',
    sai_3d_model = 'sai-3d-model'
    sai_analog_film = 'sai-analog film'
    sai_anime = 'sai-anime'
    sai_cinematic = 'sai-cinematic'
    sai_comic_book = 'sai-comic book'
    sai_ccraft_clay = 'sai-craft clay'
    sai_digital_art = 'sai-digital art'
    sai_enhance = 'sai-enhance'
    sai_fantasy_art = 'sai-fantasy art'
    sai_isometric = 'sai-isometric'
    sai_line_art = 'sai-line art'
    sai_lowpoly = 'sai-lowpoly'
    sai_neonpunk = 'sai-neonpunk'
    sai_origami = 'sai-origami'
    sai_photographic = 'sai-photographic'
    sai_pixel_art = 'sai-pixel art'
    sai_texture = 'sai-texture'
    ads_advertising = 'ads-advertising'
    ads_automotive = 'ads-automotive'
    ads_corporate = 'ads-corporate'
    ads_fashion_editorial = 'ads-fashion editorial'
    adsfood_photography = 'ads-food photography'
    ads_luxury = 'ads-luxury'
    ads_real_estate = 'ads-real estate'
    ads_retail = 'ads-retail'
    artstyle_abstract = 'artstyle-abstract'
    artstyle_abstract_expressionism = 'artstyle-abstract expressionism'
    artstyle_art_deco = 'artstyle-art deco'
    artstyle_art_nouveau = 'artstyle-art nouveau'
    artstyle_constructivist = 'artstyle-constructivist'
    artstyle_cubist = 'artstyle-cubist'
    artstyle_expressionist = 'artstyle-expressionist'
    artstyle_graffiti = 'artstyle-graffiti'
    artstyle_hyperrealism = 'artstyle-hyperrealism'
    artstyle_impressionist = 'artstyle-impressionist'
    artstyle_pointillism = 'artstyle-pointillism'
    artstyle_pop_art = 'artstyle-pop art'
    artstyle_psychedelic = 'artstyle-psychedelic'
    artstyle_renaissance = 'artstyle-renaissance'
    artstyle_steampunk = 'artstyle-steampunk'
    artstyle_surrealist = 'artstyle-surrealist'
    artstyle_typography = 'artstyle-typography'
    artstyle_watercolor = 'artstyle-watercolor'
    futuristic_biomechanical = 'futuristic-biomechanical'
    futuristic_biomechanical_cyberpunk = 'futuristic-biomechanical cyberpunk'
    futuristic_cybernetic = 'futuristic-cybernetic'
    futuristic_cybernetic_robot = 'futuristic-cybernetic robot'
    futuristic_cyberpunk_cityscape = 'futuristic-cyberpunk cityscape'
    futuristic_futuristic = 'futuristic-futuristic'
    futuristic_retro_cyberpunk = 'futuristic-retro cyberpunk'
    futuristic_retro_futurism = 'futuristic-retro futurism'
    futuristic_sci_fi = 'futuristic-sci-fi'
    futuristic_vaporwave = 'futuristic-vaporwave'
    game_bubble_bobble = 'game-bubble bobble'
    game_cyberpunk_game = 'game-cyberpunk game'
    game_fighting_game = 'game-fighting game'
    game_gta = 'game-gta'
    game_mario = 'game-mario'
    game_minecraft = 'game-minecraft'
    game_pokemon = 'game-pokemon'
    game_retro_arcade = 'game-retro arcade'
    game_retro_game = 'game-retro game'
    game_rpg_fantasy_game = 'game-rpg fantasy game'
    game_strategy_game = 'game-strategy game'
    game_streetfighter = 'game-streetfighter'
    game_zelda = 'game-zelda'
    misc_architectural = 'misc-architectural'
    misc_disco = 'misc-disco'
    misc_dreamscape = 'misc-dreamscape'
    misc_dystopian = 'misc-dystopian'
    misc_fairy_tale = 'misc-fairy tale'
    misc_gothic = 'misc-gothic'
    misc_grunge = 'misc-grunge'
    misc_horror = 'misc-horror'
    misc_kawaii = 'misc-kawaii'
    misc_lovecraftian = 'misc-lovecraftian'
    misc_macabre = 'misc-macabre'
    misc_manga = 'misc-manga'
    misc_metropolis = 'misc-metropolis'
    misc_minimalist = 'misc-minimalist'
    misc_monochrome = 'misc-monochrome'
    misc_nautical = 'misc-nautical'
    misc_space = 'misc-space'
    misc_stained_glass = 'misc-stained glass'
    misc_techwear_fashion = 'misc-techwear fashion'
    misc_tribal = 'misc-tribal'
    misc_zentangle = 'misc-zentangle'
    papercraft_collage = 'papercraft-collage'
    papercraft_flat_papercut = 'papercraft-flat papercut'
    papercraft_kirigami = 'papercraft-kirigami'
    papercraft_paper_mache = 'papercraft-paper mache'
    papercraft_paper_quilling = 'papercraft-paper quilling'
    papercraft_papercut_collage = 'papercraft-papercut collage'
    papercraft_papercut_shadow_box = 'papercraft-papercut shadow box'
    papercraft_stacked_papercut = 'papercraft-stacked papercut'
    papercraft_thick_layered_papercut = 'papercraft-thick layered papercut'
    photo_alien = 'photo-alien'
    photo_film_noir = 'photo-film noir'
    photo_hdr = 'photo-hdr'
    photo_long_exposure = 'photo-long exposure'
    photo_neon_noir = 'photo-neon noir'
    photo_silhouette = 'photo-silhouette'
    photo_tilt_shift = 'photo-tilt-shift'
    cinematic_diva = 'cinematic-diva'
    abstract_expressionism = 'Abstract Expressionism'
    academia = 'Academia'
    action_figure = 'Action Figure'
    adorable_3d_character = 'Adorable 3D Character'
    adorable_kawaii = 'Adorable Kawaii'
    art_deco = 'Art Deco'
    art_nouveau = 'Art Nouveau'
    astral_aura = 'Astral Aura'
    avant_garde = 'Avant-garde'
    baroque = 'Baroque'
    bauhaus_style_poster = 'Bauhaus-Style Poster'
    blueprint_schematic_drawing = 'Blueprint Schematic Drawing'
    caricature = 'Caricature'
    cel_shaded_art = 'Cel Shaded Art'
    character_design_sheet = 'Character Design Sheet'
    classicism_art = 'Classicism Art'
    color_field_painting = 'Color Field Painting'
    colored_pencil_art = 'Colored Pencil Art'
    conceptual_art = 'Conceptual Art'
    constructivism = 'Constructivism'
    cubism = 'Cubism'
    dadaism = 'Dadaism'
    dark_fantasy = 'Dark Fantasy'
    dark_moody_atmosphere = 'Dark Moody Atmosphere'
    dmt_art_style = 'DMT Art Style'
    doodle_art = 'Doodle Art'
    double_exposure = 'Double Exposure'
    dripping_paint_splatter_art = 'Dripping Paint Splatter Art'
    expressionism = 'Expressionism'
    faded_polaroid_photo = 'Faded Polaroid Photo'
    fauvism = 'Fauvism'
    flat_2d_art = 'Flat 2D Art'
    fortnite_art_style = 'Fortnite Art Style'
    futurism = 'Futurism'
    glitchcore = 'Glitchcore'
    glo_fi = 'Glo-fi'
    googie_art_style = 'Googie Art Style'
    graffiti_art = 'Graffiti Art'
    harlem_renaissance_art = 'Harlem Renaissance Art'
    high_fashion = 'High Fashion'
    idyllic = 'Idyllic'
    impressionism = 'Impressionism'
    infographic_drawing = 'Infographic Drawing'
    ink_dripping_drawing = 'Ink Dripping Drawing'
    japanese_ink_drawing = 'Japanese Ink Drawing'
    knolling_photography = 'Knolling Photography'
    light_cheery_atmosphere = 'Light Cheery Atmosphere'
    logo_design = 'Logo Design'
    luxurious_elegance = 'Luxurious Elegance'
    macro_photography = 'Macro Photography'
    mandola_art = 'Mandola Art'
    marker_drawing = 'Marker Drawing'
    medievalism = 'Medievalism'
    minimalism = 'Minimalism'
    neo_baroque = 'Neo-Baroque'
    neo_byzantine = 'Neo-Byzantine'
    neo_futurism = 'Neo-Futurism'
    neo_impressionism = 'Neo-Impressionism'
    neo_rococo = 'Neo-Rococo'
    neoclassicism = 'Neoclassicism'
    op_art = 'Op Art'
    ornate_and_intricate = 'Ornate and Intricate'
    pencil_sketch_drawing = 'Pencil Sketch Drawing'
    pop_art_2 = 'Pop Art 2'
    rococo = 'Rococo'
    silhouette_art = 'Silhouette Art'
    simple_vector_art = 'Simple Vector Art'
    sketchup = 'Sketchup'
    steampunk_2 = 'Steampunk 2'
    surrealism = 'Surrealism'
    suprematism = 'Suprematism'
    terragen = 'Terragen'
    tranquil_relaxing_atmosphere = 'Tranquil Relaxing Atmosphere'
    sticker_designs = 'Sticker Designs'
    vibrant_rim_light = 'Vibrant Rim Light'
    volumetric_lighting = 'Volumetric Lighting'
    watercolor_2 = 'Watercolor 2'
    whimsical_and_playful = 'Whimsical and Playful'

class AspectRatio(str, Enum):
    a_0_5 = '704×1408'
    a_0_52 = '704×1344'
    a_0_57 = '768×1344'
    a_0_6 = '768×1280'
    a_0_68 = '832×1216'
    a_0_72 = '832×1152'
    a_0_78 = '896×1152'
    a_0_82 = '896×1088'
    a_0_88 = '960×1088'
    a_0_94 = '960×1024'
    a_1_0 = '1024×1024'
    a_1_07 = '1024×960'
    a_1_13 = '1088×960'
    a_1_21 = '1088×896'
    a_1_29 = '1152×896'
    a_1_38 = '1152×832'
    a_1_46 = '1216×832'
    a_1_67 = '1280×768'
    a_1_75 = '1344×768'
    a_1_91 = '1344×704'
    a_2_0 = '1408×704'
    a_2_09 = '1472×704'
    a_2_4 = '1536×640'
    a_2_5 = '1600×640'
    a_2_89 = '1664×576'
    a_3_0 = '1728×576'

class Text2ImgRequest(BaseModel):
    prompt: str = ''
    negative_promit: str = ''
    style_selections: List[FooocusStyle] = [FooocusStyle.default]
    performance_selection: PerfomanceSelection = PerfomanceSelection.speed
    aspect_ratios_selection: AspectRatio = AspectRatio.a_1_29
    image_number: int = Field(default=2, description="Image number", min=1)
    image_seed: int  | None = None
    sharpness: float = Field(default=2.0, min=0.0, max=30.0)
    base_model_name: str = 'sd_xl_base_1.0_0.9vae.safetensors'
    refiner_model_name: str = 'sd_xl_refiner_1.0_0.9vae.safetensors'
    loras: List[Lora] = [Lora(model_name='sd_xl_offset_example-lora_1.0.safetensors', weight=0.5)]

@app.post("/v1/generation/text-to-image", responses = {
    200: {
        "description": "PNG bytes if request's 'Accept' header is 'image/png', otherwise JSON",
        "content": {
            "application/json": {
                "example": {
                    "base64": ["...very long string..."],
                    "seed": 1050625087,
                    "finishReason": "SUCCESS"
                }
            },
            "image/png": {
                "example": "PNG bytes, what did you expect?"
            }
        }
    }
})
def text2img_generation(req: Text2ImgRequest, accept: Annotated[str | None,  Header] = None):
    import modules.default_pipeline as pipeline
    import modules.patch as patch
    import modules.virtual_memory as virtual_memory
    import comfy.model_management as model_management
    from modules.util import join_prompts, remove_empty_str
    from modules.private_logger import log
    from fooocusapi.api_utils import narray_to_base64img

    if accept == 'image/png':
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    loras = [(l.model_name, l.weight) for l in req.loras]
    loras_user_raw_input = copy.deepcopy(loras)


    style_selections = [s.value for s in req.style_selections]
    raw_style_selections = copy.deepcopy(style_selections)
    if fooocus_expansion in style_selections:
        use_expansion = True
        style_selections.remove(fooocus_expansion)
    else:
        use_expansion = False
    
    use_style = len(req. style_selections) > 0
    patch.sharpness = req.sharpness
    patch.negative_adm = True
    initial_latent = None
    denoising_strength = 1.0
    tiled = False
    
    if req.performance_selection == PerfomanceSelection.speed:
        steps = 30
        switch = 20
    else:
        steps = 60
        switch = 40

    pipeline.clear_all_caches()
    width, height = aspect_ratios[req.aspect_ratios_selection.value]

    raw_prompt = req.prompt
    raw_negative_prompt = req.negative_promit

    prompts = remove_empty_str([safe_str(p) for p in req.prompt.split('\n')], default='')
    negative_prompts = remove_empty_str([safe_str(p) for p in req.negative_promit.split('\n')], default='')

    prompt = prompts[0]
    negative_prompt = negative_prompts[0]

    extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
    extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

    seed = req.image_seed
    max_seed = int(1024 * 1024 * 1024)
    if not isinstance(seed, int):
        seed = random.randint(1, max_seed)
    if seed < 0:
        seed = - seed
    seed = seed % max_seed

    pipeline.refresh_everything(
        refiner_model_name=req.refiner_model_name,
        base_model_name=req.base_model_name,
        loras=loras
    )

    positive_basic_workloads = []
    negative_basic_workloads = []

    if use_style:
        for s in style_selections:
            p, n = apply_style(s, positive=prompt)
            positive_basic_workloads.append(p)
            negative_basic_workloads.append(n)
    else:
        positive_basic_workloads.append(prompt)

    positive_basic_workloads = positive_basic_workloads + extra_positive_prompts
    negative_basic_workloads = negative_basic_workloads + extra_negative_prompts

    positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=prompt)
    negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=negative_prompt)

    positive_top_k = len(positive_basic_workloads)
    negative_top_k = len(negative_basic_workloads)

    tasks = [dict(
        task_seed=seed + i,
        positive=positive_basic_workloads,
        negative=negative_basic_workloads,
        expansion='',
        c=[None, None],
        uc=[None, None],
    ) for i in range(req.image_number)]

    if use_expansion:
        for i, t in enumerate(tasks):
            expansion = pipeline.expansion(prompt, t['task_seed'])
            print(f'[Prompt Expansion] New suffix: {expansion}')
            t['expansion'] = expansion
            t['positive'] = copy.deepcopy(t['positive']) + [join_prompts(prompt, expansion)]  # Deep copy.

    for i, t in enumerate(tasks):
        t['c'][0] = pipeline.clip_encode(sd=pipeline.xl_base_patched, texts=t['positive'],
                                            pool_top_k=positive_top_k)
        
    for i, t in enumerate(tasks):
        t['uc'][0] = pipeline.clip_encode(sd=pipeline.xl_base_patched, texts=t['negative'],
                                            pool_top_k=negative_top_k)

    if pipeline.xl_refiner is not None:
        virtual_memory.load_from_virtual_memory(pipeline.xl_refiner.clip.cond_stage_model)

        for i, t in enumerate(tasks):
            t['c'][1] = pipeline.clip_encode(sd=pipeline.xl_refiner, texts=t['positive'],
                                                pool_top_k=positive_top_k)

        for i, t in enumerate(tasks):
            t['uc'][1] = pipeline.clip_encode(sd=pipeline.xl_refiner, texts=t['negative'],
                                                pool_top_k=negative_top_k)

        virtual_memory.try_move_to_virtual_memory(pipeline.xl_refiner.clip.cond_stage_model)

    results = []
    all_steps = steps * req.image_number

    def callback(step, x0, x, total_steps, y):
        done_steps = current_task_id * steps + step
        print(f"Finished {done_steps}/{all_steps}")

    print(f'[ADM] Negative ADM = {patch.negative_adm}')

    for current_task_id, task in enumerate(tasks):
        try:
            imgs = pipeline.process_diffusion(
                positive_cond=task['c'],
                negative_cond=task['uc'],
                steps=steps,
                switch=switch,
                width=width,
                height=height,
                image_seed=task['task_seed'],
                callback=callback,
                latent=initial_latent,
                denoise=denoising_strength,
                tiled=tiled
            )

            for x in imgs:
                d = [
                    ('Prompt', raw_prompt),
                    ('Negative Prompt', raw_negative_prompt),
                    ('Fooocus V2 Expansion', task['expansion']),
                    ('Styles', str(raw_style_selections)),
                    ('Performance', req.performance_selection),
                    ('Resolution', str((width, height))),
                    ('Sharpness', req.sharpness),
                    ('Base Model', req.base_model_name),
                    ('Refiner Model', req.refiner_model_name),
                    ('Seed', task['task_seed'])
                ]
                for n, w in loras_user_raw_input:
                    if n != 'None':
                        d.append((f'LoRA [{n}] weight', w))
                log(x, d, single_line_number=3)

            results += imgs
        except model_management.InterruptProcessingException as e:
            print('User stopped')
            break


    if streaming_output:
        bytes = narray_to_bytesimg(results[0])
        return Response(bytes, media_type='image/png')
    else:
        results = [narray_to_base64img(narr) for narr in results]
        return {"base64": results, "seed": task['task_seed'], "finishReason": "SUCCESS"}

def start_app(args):
    uvicorn.run("fooocusapi.api:app", host=args.host, port=args.port, log_level=args.log_level)
