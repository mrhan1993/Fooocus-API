from fastapi import Form, UploadFile
from fastapi.params import File
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from typing import List
from enum import Enum

from pydantic_core import InitErrorDetails
from fooocusapi.parameters import GenerationFinishReason
import modules.flags as flags


class Lora(BaseModel):
    model_name: str
    weight: float = Field(default=0.5, min=-2, max=2)

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )


class PerfomanceSelection(str, Enum):
    speed = 'Speed'
    quality = 'Quality'


class FooocusStyle(str, Enum):
    fooocus_expansion = 'Fooocus V2'
    default = 'Default (Slightly Cinematic)'
    fooocus_anime = 'Fooocus Anime'
    fooocus_realistic = 'Fooocus Realistic'
    fooocus_strong_negative = 'Fooocus Strong Negative'
    sai_3d_model = 'SAI 3D Model'
    sai_analog_film = 'SAI Analog Film'
    sai_anime = 'SAI Anime'
    sai_cinematic = 'SAI Cinematic'
    sai_comic_book = 'SAI Comic Book'
    sai_ccraft_clay = 'SAI Craft Clay'
    sai_digital_art = 'SAI Digital Art'
    sai_enhance = 'SAI Enhance'
    sai_fantasy_art = 'SAI Fantasy Art'
    sai_isometric = 'SAI Isometric'
    sai_line_art = 'SAI Line Art'
    sai_lowpoly = 'SAI Lowpoly'
    sai_neonpunk = 'SAI Neonpunk'
    sai_origami = 'SAI Prigami'
    sai_photographic = 'SAI Photographic'
    sai_pixel_art = 'SAI Pixel Art'
    sai_texture = 'SAI Texture'
    ads_advertising = 'Ads Advertising'
    ads_automotive = 'Ads Automotive'
    ads_corporate = 'Ads Corporate'
    ads_fashion_editorial = 'Ads Fashion Editorial'
    adsfood_photography = 'Ads Food Photography'
    ads_luxury = 'Ads Luxury'
    ads_real_estate = 'Ads Real Estate'
    ads_retail = 'Ads Retail'
    artstyle_abstract = 'Artstyle Abstract'
    artstyle_abstract_expressionism = 'Artstyle Abstract Expressionism'
    artstyle_art_deco = 'Artstyle Art Deco'
    artstyle_art_nouveau = 'Artstyle Art Nouveau'
    artstyle_constructivist = 'Artstyle Constructivist'
    artstyle_cubist = 'Artstyle Cubist'
    artstyle_expressionist = 'Artstyle Expressionist'
    artstyle_graffiti = 'Artstyle Graffiti'
    artstyle_hyperrealism = 'Artstyle Hyperrealism'
    artstyle_impressionist = 'Artstyle Impressionist'
    artstyle_pointillism = 'Artstyle Pointillism'
    artstyle_pop_art = 'Artstyle Pop Art'
    artstyle_psychedelic = 'Artstyle Psychedelic'
    artstyle_renaissance = 'Artstyle Renaissance'
    artstyle_steampunk = 'Artstyle Steampunk'
    artstyle_surrealist = 'Artstyle Surrealist'
    artstyle_typography = 'Artstyle Typography'
    artstyle_watercolor = 'Artstyle Watercolor'
    futuristic_biomechanical = 'Futuristic Biomechanical'
    futuristic_biomechanical_cyberpunk = 'Futuristic Biomechanical Cyberpunk'
    futuristic_cybernetic = 'Futuristic Cybernetic'
    futuristic_cybernetic_robot = 'Futuristic Cybernetic Robot'
    futuristic_cyberpunk_cityscape = 'Futuristic Cyberpunk Cityscape'
    futuristic_futuristic = 'Futuristic Futuristic'
    futuristic_retro_cyberpunk = 'Futuristic Retro Cyberpunk'
    futuristic_retro_futurism = 'Futuristic Retro Futurism'
    futuristic_sci_fi = 'Futuristic Sci Fi'
    futuristic_vaporwave = 'Futuristic Vaporwave'
    game_bubble_bobble = 'Game Bubble Bobble'
    game_cyberpunk_game = 'Game Cyberpunk Game'
    game_fighting_game = 'Game Fighting Game'
    game_gta = 'Game Gta'
    game_mario = 'Game Mario'
    game_minecraft = 'Game Minecraft'
    game_pokemon = 'Game Pokemon'
    game_retro_arcade = 'Game Retro Arcade'
    game_retro_game = 'Game Retro Game'
    game_rpg_fantasy_game = 'Game Rpg Fantasy Game'
    game_strategy_game = 'Game Strategy Game'
    game_streetfighter = 'Game Streetfighter'
    game_zelda = 'Game Zelda'
    misc_architectural = 'Misc Architectural'
    misc_disco = 'Misc Disco'
    misc_dreamscape = 'Misc Dreamscape'
    misc_dystopian = 'Misc Dystopian'
    misc_fairy_tale = 'Misc Fairy Tale'
    misc_gothic = 'Misc Gothic'
    misc_grunge = 'Misc Grunge'
    misc_horror = 'Misc Horror'
    misc_kawaii = 'Misc Kawaii'
    misc_lovecraftian = 'Misc Lovecraftian'
    misc_macabre = 'Misc Macabre'
    misc_manga = 'Misc Manga'
    misc_metropolis = 'Misc Metropolis'
    misc_minimalist = 'Misc Minimalist'
    misc_monochrome = 'Misc Monochrome'
    misc_nautical = 'Misc Nautical'
    misc_space = 'Misc Space'
    misc_stained_glass = 'Misc Stained Glass'
    misc_techwear_fashion = 'Misc Techwear Fashion'
    misc_tribal = 'Misc Tribal'
    misc_zentangle = 'Misc Zentangle'
    papercraft_collage = 'Papercraft Collage'
    papercraft_flat_papercut = 'Papercraft Flat Papercut'
    papercraft_kirigami = 'Papercraft Kirigami'
    papercraft_paper_mache = 'Papercraft Paper Mache'
    papercraft_paper_quilling = 'Papercraft Paper Quilling'
    papercraft_papercut_collage = 'Papercraft Papercut Collage'
    papercraft_papercut_shadow_box = 'Papercraft Papercut Shadow Box'
    papercraft_stacked_papercut = 'Papercraft Stacked Papercut'
    papercraft_thick_layered_papercut = 'Papercraft Thick Layered Papercut'
    photo_alien = 'Photo Alien'
    photo_film_noir = 'Photo Film Noir'
    photo_hdr = 'Photo Hdr'
    photo_long_exposure = 'Photo Long Exposure'
    photo_neon_noir = 'Photo Neon Noir'
    photo_silhouette = 'Photo Silhouette'
    photo_tilt_shift = 'Photo Tilt Shift'
    cinematic_diva = 'Cinematic Diva'
    abstract_expressionism = 'Abstract Expressionism'
    academia = 'Academia'
    action_figure = 'Action Figure'
    adorable_3d_character = 'Adorable 3D Character'
    adorable_kawaii = 'Adorable Kawaii'
    art_deco = 'Art Deco'
    art_nouveau = 'Art Nouveau'
    astral_aura = 'Astral Aura'
    avant_garde = 'Avant Garde'
    baroque = 'Baroque'
    bauhaus_style_poster = 'Bauhaus Style Poster'
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
    faded_polaroid_Photo = 'Faded Polaroid Photo'
    fauvism = 'Fauvism'
    flat_2d_art = 'Flat 2D Art'
    fortnite_art_style = 'Fortnite Art Style'
    futurism = 'Futurism'
    glitchcore = 'Glitchcore'
    glo_fi = 'Glo Fi'
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
    neo_baroque = 'Neo Baroque'
    neo_byzantine = 'Neo Byzantine'
    neo_futurism = 'Neo Futurism'
    neo_impressionism = 'Neo Impressionism'
    neo_rococo = 'Neo Rococo'
    neoclassicism = 'Neoclassicism'
    op_art = 'Op Art'
    ornate_and_intricate = 'Ornate And Intricate'
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
    whimsical_and_playful = 'Whimsical And Playful'


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


class UpscaleOrVaryMethod(str, Enum):
    subtle_variation = 'Vary (Subtle)'
    strong_variation = 'Vary (Strong)'
    upscale_15 = 'Upscale (1.5x)'
    upscale_2 = 'Upscale (2x)'
    upscale_fast = 'Upscale (Fast 2x)'


class OutpaintExpansion(str, Enum):
    left = 'Left'
    right = 'Right'
    top = 'Top'
    bottom = 'Bottom'


class ControlNetType(str, Enum):
    cn_ip = 'Image Prompt'
    cn_canny = 'PyraCanny'
    cn_cpds = 'CPDS'


class ImagePrompt(BaseModel):
    cn_img: UploadFile | None = Field(default=None)
    cn_stop: float = Field(default=0.4, min=0, max=1)
    cn_weight: float | None = Field(
        default=None, min=0, max=2, description="None for default value")
    cn_type: ControlNetType = Field(default=ControlNetType.cn_ip)


class Text2ImgRequest(BaseModel):
    prompt: str = ''
    negative_prompt: str = ''
    style_selections: List[FooocusStyle] = [
        FooocusStyle.fooocus_expansion, FooocusStyle.default]
    performance_selection: PerfomanceSelection = PerfomanceSelection.speed
    aspect_ratios_selection: AspectRatio = AspectRatio.a_1_29
    image_number: int = Field(
        default=1, description="Image number", min=1, max=32)
    image_seed: int = Field(default=-1, description="Seed to generate image, -1 for random")
    sharpness: float = Field(default=2.0, min=0.0, max=30.0)
    guidance_scale: float = Field(default=7.0, min=1.0, max=30.0)
    base_model_name: str = 'sd_xl_base_1.0_0.9vae.safetensors'
    refiner_model_name: str = 'sd_xl_refiner_1.0_0.9vae.safetensors'
    loras: List[Lora] = Field(default=[
        Lora(model_name='sd_xl_offset_example-lora_1.0.safetensors', weight=0.5)])


class ImgUpscaleOrVaryRequest(Text2ImgRequest):
    input_image: UploadFile
    uov_method: UpscaleOrVaryMethod

    @classmethod
    def as_form(cls, input_image: UploadFile = Form(description="Init image for upsacale or outpaint"),
                uov_method: UpscaleOrVaryMethod = Form(),
                prompt: str = Form(''),
                negative_prompt: str = Form(''),
                style_selections: List[str] = Form([
                    FooocusStyle.fooocus_expansion, FooocusStyle.default], description="Fooocus style selections, seperated by comma"),
                performance_selection: PerfomanceSelection = Form(
                    PerfomanceSelection.speed),
                aspect_ratios_selection: AspectRatio = Form(
                    AspectRatio.a_1_29),
                image_number: int = Form(
                    default=1, description="Image number", ge=1, le=32),
                image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
                sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
                guidance_scale: float = Form(default=7.0, ge=1.0, le=30.0),
                base_model_name: str = Form(
                    'sd_xl_base_1.0_0.9vae.safetensors'),
                refiner_model_name: str = Form(
                    'sd_xl_refiner_1.0_0.9vae.safetensors'),
                l1: str | None = Form(
                    'sd_xl_offset_example-lora_1.0.safetensors'),
                w1: float = Form(default=0.5, ge=-2, le=2),
                l2: str | None = Form(None),
                w2: float = Form(default=0.5, ge=-2, le=2),
                l3: str | None = Form(None),
                w3: float = Form(default=0.5, ge=-2, le=2),
                l4: str | None = Form(None),
                w4: float = Form(default=0.5, ge=-2, le=2),
                l5: str | None = Form(None),
                w5: float = Form(default=0.5, ge=-2, le=2),
                ):
        style_selection_arr: List[FooocusStyle] = []
        for part in style_selections:
            if len(part) > 0:
                for s in part.split(','):
                    try:
                        style = FooocusStyle(s)
                        style_selection_arr.append(style)
                    except ValueError as ve:
                        err = InitErrorDetails(type='enum', loc=['style_selections'], input=style_selections, ctx={
                            'expected': 'Valid fooocus styles seperated by comma'})
                        raise RequestValidationError(errors=[err])

        loras: List[Lora] = []
        lora_config = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]
        for config in lora_config:
            lora_model, lora_weight = config
            if lora_model is not None and len(lora_model) > 0:
                loras.append(Lora(model_name=lora_model, weight=lora_weight))

        return cls(input_image=input_image, uov_method=uov_method, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name,
                   loras=loras)


class ImgInpaintOrOutpaintRequest(Text2ImgRequest):
    input_image: UploadFile
    input_mask: UploadFile | None
    outpaint_selections: List[OutpaintExpansion]

    @classmethod
    def as_form(cls, input_image: UploadFile = Form(description="Init image for inpaint or outpaint"),
                input_mask: UploadFile = Form(
                    File(None), description="Inpaint or outpaint mask"),
                outpaint_selections: List[str] = Form(
                    [], description="Outpaint expansion selections, literal 'Left', 'Right', 'Top', 'Bottom' seperated by comma"),
                prompt: str = Form(''),
                negative_prompt: str = Form(''),
                style_selections: List[str] = Form([
                    FooocusStyle.fooocus_expansion, FooocusStyle.default], description="Fooocus style selections, seperated by comma"),
                performance_selection: PerfomanceSelection = Form(
                    PerfomanceSelection.speed),
                aspect_ratios_selection: AspectRatio = Form(
                    AspectRatio.a_1_29),
                image_number: int = Form(
                    default=1, description="Image number", ge=1, le=32),
                image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
                sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
                guidance_scale: float = Form(default=7.0, ge=1.0, le=30.0),
                base_model_name: str = Form(
                    'sd_xl_base_1.0_0.9vae.safetensors'),
                refiner_model_name: str = Form(
                    'sd_xl_refiner_1.0_0.9vae.safetensors'),
                l1: str | None = Form(
                    'sd_xl_offset_example-lora_1.0.safetensors'),
                w1: float = Form(default=0.5, ge=-2, le=2),
                l2: str | None = Form(None),
                w2: float = Form(default=0.5, ge=-2, le=2),
                l3: str | None = Form(None),
                w3: float = Form(default=0.5, ge=-2, le=2),
                l4: str | None = Form(None),
                w4: float = Form(default=0.5, ge=-2, le=2),
                l5: str | None = Form(None),
                w5: float = Form(default=0.5, ge=-2, le=2),
                ):

        if isinstance(input_mask, File):
            input_mask = None
        
        outpaint_selections_arr: List[OutpaintExpansion] = []
        for part in outpaint_selections:
            if len(part) > 0:
                for s in part.split(','):
                    try:
                        expansion = OutpaintExpansion(s)
                        outpaint_selections_arr.append(expansion)
                    except ValueError as ve:
                        err = InitErrorDetails(type='enum', loc=['outpaint_selections'], input=outpaint_selections, ctx={
                            'expected': "Literal 'Left', 'Right', 'Top', 'Bottom' seperated by comma"})
                        raise RequestValidationError(errors=[err])

        style_selection_arr: List[FooocusStyle] = []
        for part in style_selections:
            if len(part) > 0:
                for s in part.split(','):
                    try:
                        expansion = FooocusStyle(s)
                        style_selection_arr.append(expansion)
                    except ValueError as ve:
                        err = InitErrorDetails(type='enum', loc=['style_selections'], input=style_selections, ctx={
                            'expected': 'Valid fooocus styles seperated by comma'})
                        raise RequestValidationError(errors=[err])

        loras: List[Lora] = []
        lora_config = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]
        for config in lora_config:
            lora_model, lora_weight = config
            if lora_model is not None and len(lora_model) > 0:
                loras.append(Lora(model_name=lora_model, weight=lora_weight))

        return cls(input_image=input_image, input_mask=input_mask, outpaint_selections=outpaint_selections_arr, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name,
                   loras=loras)


class ImgPromptRequest(Text2ImgRequest):
    image_prompts: List[ImagePrompt]

    @classmethod
    def as_form(cls, cn_img1: UploadFile = Form(File(None), description="Input image for image prompt"),
                cn_stop1: float | None = Form(
                    default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
                cn_weight1: float | None = Form(
                    default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
                cn_type1: ControlNetType = Form(
                    default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
                cn_img2: UploadFile = Form(
                    File(None), description="Input image for image prompt"),
                cn_stop2: float | None = Form(
                    default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
                cn_weight2: float | None = Form(
                    default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
                cn_type2: ControlNetType = Form(
                    default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
                cn_img3: UploadFile = Form(
                    File(None), description="Input image for image prompt"),
                cn_stop3: float | None = Form(
                    default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
                cn_weight3: float | None = Form(
                    default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
                cn_type3: ControlNetType = Form(
                    default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
                cn_img4: UploadFile = Form(
                    File(None), description="Input image for image prompt"),
                cn_stop4: float | None = Form(
                    default=None, ge=0, le=1, description="Stop at for image prompt, None for default value"),
                cn_weight4: float | None = Form(
                    default=None, ge=0, le=2, description="Weight for image prompt, None for default value"),
                cn_type4: ControlNetType = Form(
                    default=ControlNetType.cn_ip, description="ControlNet type for image prompt"),
                prompt: str = Form(''),
                negative_prompt: str = Form(''),
                style_selections: List[str] = Form([
                    FooocusStyle.fooocus_expansion, FooocusStyle.default], description="Fooocus style selections, seperated by comma"),
                performance_selection: PerfomanceSelection = Form(
                    PerfomanceSelection.speed),
                aspect_ratios_selection: AspectRatio = Form(
                    AspectRatio.a_1_29),
                image_number: int = Form(
                    default=1, description="Image number", ge=1, le=32),
                image_seed: int = Form(default=-1, description="Seed to generate image, -1 for random"),
                sharpness: float = Form(default=2.0, ge=0.0, le=30.0),
                guidance_scale: float = Form(default=7.0, ge=1.0, le=30.0),
                base_model_name: str = Form(
                    'sd_xl_base_1.0_0.9vae.safetensors'),
                refiner_model_name: str = Form(
                    'sd_xl_refiner_1.0_0.9vae.safetensors'),
                l1: str | None = Form(
                    'sd_xl_offset_example-lora_1.0.safetensors'),
                w1: float = Form(default=0.5, ge=-2, le=2),
                l2: str | None = Form(None),
                w2: float = Form(default=0.5, ge=-2, le=2),
                l3: str | None = Form(None),
                w3: float = Form(default=0.5, ge=-2, le=2),
                l4: str | None = Form(None),
                w4: float = Form(default=0.5, ge=-2, le=2),
                l5: str | None = Form(None),
                w5: float = Form(default=0.5, ge=-2, le=2),
                ):
        if isinstance(cn_img1, File):
            cn_img1 = None
        if isinstance(cn_img2, File):
            cn_img2 = None
        if isinstance(cn_img3, File):
            cn_img3 = None
        if isinstance(cn_img4, File):
            cn_img4 = None

        image_prompts: List[ImagePrompt] = []
        image_prompt_config = [(cn_img1, cn_stop1, cn_weight1, cn_type1), (cn_img2, cn_stop2, cn_weight2, cn_type2),
                               (cn_img3, cn_stop3, cn_weight3, cn_type3), (cn_img4, cn_stop4, cn_weight4, cn_type4)]
        for config in image_prompt_config:
            cn_img, cn_stop, cn_weight, cn_type = config
            if cn_stop is None:
                cn_stop = flags.default_parameters[cn_type.value][0]
            if cn_weight is None:
                cn_weight = flags.default_parameters[cn_type.value][1]
            image_prompts.append(ImagePrompt(
                cn_img=cn_img, cn_stop=cn_stop, cn_weight=cn_weight, cn_type=cn_type))

        style_selection_arr: List[FooocusStyle] = []
        for part in style_selections:
            if len(part) > 0:
                for s in part.split(','):
                    try:
                        expansion = FooocusStyle(s)
                        style_selection_arr.append(expansion)
                    except ValueError as ve:
                        err = InitErrorDetails(type='enum', loc=['style_selections'], input=style_selections, ctx={
                            'expected': 'Valid fooocus styles seperated by comma'})
                        raise RequestValidationError(errors=[err])

        loras: List[Lora] = []
        lora_config = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]
        for config in lora_config:
            lora_model, lora_weight = config
            if lora_model is not None and len(lora_model) > 0:
                loras.append(Lora(model_name=lora_model, weight=lora_weight))

        return cls(image_prompts=image_prompts, prompt=prompt, negative_prompt=negative_prompt, style_selections=style_selection_arr,
                   performance_selection=performance_selection, aspect_ratios_selection=aspect_ratios_selection,
                   image_number=image_number, image_seed=image_seed, sharpness=sharpness, guidance_scale=guidance_scale,
                   base_model_name=base_model_name, refiner_model_name=refiner_model_name,
                   loras=loras)


class GeneratedImageBase64(BaseModel):
    base64: str | None = Field(
        description="Image encoded in base64, or null if finishReasen is not 'SUCCESS'")
    seed: int = Field(description="The seed associated with this image")
    finish_reason: GenerationFinishReason
