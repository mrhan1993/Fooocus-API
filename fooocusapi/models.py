from pydantic import BaseModel, ConfigDict, Field
from typing import List
from enum import Enum


class Lora(BaseModel):
    model_name: str
    weight: float

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )


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
    image_seed: int | None = None
    sharpness: float = Field(default=2.0, min=0.0, max=30.0)
    guidance_scale: float = Field(default=7.0, min=1.0, max=30.0)
    base_model_name: str = 'sd_xl_base_1.0_0.9vae.safetensors'
    refiner_model_name: str = 'sd_xl_refiner_1.0_0.9vae.safetensors'
    loras: List[Lora] = [
        Lora(model_name='sd_xl_offset_example-lora_1.0.safetensors', weight=0.5)]


class UpscaleOrVaryMethod(str, Enum):
    subtle_variation = 'Vary (Subtle)'
    strong_variation = 'Vary (Strong)'
    upscale_15 = 'Upscale (1.5x)'
    upscale_2 = 'Upscale (2x)'
    upscale_fast = 'Upscale (Fast 2x)'


class ImgUpscaleOrVaryRequest(Text2ImgRequest):
    uov_method: UpscaleOrVaryMethod = UpscaleOrVaryMethod.subtle_variation


class GenerationFinishReason(str, Enum):
    success = 'SUCCESS'
    queue_is_full = 'QUEUE_IS_FULL'
    user_cancel = 'USER_CANCEL'
    error = 'ERROR'


class GeneratedImage(BaseModel):
    im: object | None
    seed: int
    finish_reason: GenerationFinishReason


class GeneratedImageBase64(BaseModel):
    base64: str | None = Field(
        description="Image encoded in base64, or null if finishReasen is not 'SUCCESS'")
    seed: int = Field(description="The seed associated with this image")
    finish_reason: GenerationFinishReason


class TaskType(str, Enum):
    text2img = 'text2img'


class QueueTask(object):
    is_finished: bool = False
    start_millis: int = 0
    finish_millis: int = 0
    finish_with_error: bool = False
    task_result: any = None

    def __init__(self, seq: int, type: TaskType, req_param: dict, in_queue_millis: int):
        self.seq = seq
        self.type = type
        self.req_param = req_param
        self.in_queue_millis = in_queue_millis
