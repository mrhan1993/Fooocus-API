import copy
import random
import time
from typing import Annotated, List
from fastapi import FastAPI, Header, Response
import uvicorn
from fooocusapi.api_utils import narray_to_bytesimg
from fooocusapi.models import GeneratedImageItem, GenerationFinishReason, PerfomanceSelection, QueueTask, TaskType, Text2ImgRequest
from fooocusapi.task_queue import TaskQueue
from modules.expansion import safe_str
from modules.sdxl_styles import apply_style, fooocus_expansion, aspect_ratios

app = FastAPI()

task_queue = TaskQueue()


@app.post("/v1/generation/text-to-image", response_model=List[GeneratedImageItem], responses={
    200: {
        "description": "PNG bytes if request's 'Accept' header is 'image/png', otherwise JSON",
        "content": {
            "application/json": {
                "example": [{
                    "base64": "...very long string...",
                    "seed": 1050625087,
                    "finish_reason": "SUCCESS"
                }]
            },
            "image/png": {
                "example": "PNG bytes, what did you expect?"
            }
        }
    }
})
def text2img_generation(req: Text2ImgRequest, accept: Annotated[str | None,  Header] = None):
    print("text2img_generation")
    import modules.default_pipeline as pipeline
    import modules.patch as patch
    import modules.virtual_memory as virtual_memory
    import comfy.model_management as model_management
    from modules.util import join_prompts, remove_empty_str
    from modules.private_logger import log
    from fooocusapi.api_utils import narray_to_base64img

    print("text2img_generation start")

    if accept == 'image/png':
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    task_seq = task_queue.add_task(TaskType.text2img, {
                        'body': req.__dict__, 'accept': accept})
    if task_seq is None:
        print("[Task Queue] The task queue has reached limit")
        results = []
        for i in range(0, req.image_number):
            results.append({'im': None, 'seed': 0, 'finish_reason': GenerationFinishReason.queue_is_full})
        return results
    
    sleep_seconds = 0
    while not task_queue.is_task_ready_to_start(task_seq):
        if sleep_seconds == 0:
            print(f"[Task Queue] Waiting for task queue become free, seq={task_seq}")

        time.sleep(1)
        sleep_seconds += 1
        if sleep_seconds % 10 == 0:
            print(f"[Task Queue] Already waiting for {sleep_seconds}S, seq={task_seq}")

    print(f"[Task Queue] Task queue is free, start task, seq={task_seq}")

    task_queue.start_task(task_seq)

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

    prompts = remove_empty_str([safe_str(p)
                               for p in req.prompt.split('\n')], default='')
    negative_prompts = remove_empty_str(
        [safe_str(p) for p in req.negative_promit.split('\n')], default='')

    prompt = prompts[0]
    negative_prompt = negative_prompts[0]

    extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
    extra_negative_prompts = negative_prompts[1:] if len(
        negative_prompts) > 1 else []

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

    positive_basic_workloads = remove_empty_str(
        positive_basic_workloads, default=prompt)
    negative_basic_workloads = remove_empty_str(
        negative_basic_workloads, default=negative_prompt)

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
            # Deep copy.
            t['positive'] = copy.deepcopy(
                t['positive']) + [join_prompts(prompt, expansion)]

    for i, t in enumerate(tasks):
        t['c'][0] = pipeline.clip_encode(sd=pipeline.xl_base_patched, texts=t['positive'],
                                         pool_top_k=positive_top_k)

    for i, t in enumerate(tasks):
        t['uc'][0] = pipeline.clip_encode(sd=pipeline.xl_base_patched, texts=t['negative'],
                                          pool_top_k=negative_top_k)

    if pipeline.xl_refiner is not None:
        virtual_memory.load_from_virtual_memory(
            pipeline.xl_refiner.clip.cond_stage_model)

        for i, t in enumerate(tasks):
            t['c'][1] = pipeline.clip_encode(sd=pipeline.xl_refiner, texts=t['positive'],
                                             pool_top_k=positive_top_k)

        for i, t in enumerate(tasks):
            t['uc'][1] = pipeline.clip_encode(sd=pipeline.xl_refiner, texts=t['negative'],
                                              pool_top_k=negative_top_k)

        virtual_memory.try_move_to_virtual_memory(
            pipeline.xl_refiner.clip.cond_stage_model)

    results = []
    all_steps = steps * req.image_number

    def callback(step, x0, x, total_steps, y):
        done_steps = current_task_id * steps + step
        print(f"Finished {done_steps}/{all_steps}")

    print(f'[ADM] Negative ADM = {patch.negative_adm}')

    process_with_error = False
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

            results.append({'im': imgs[0], 'seed': task['task_seed'],
                           'finish_reason': GenerationFinishReason.success})
        except model_management.InterruptProcessingException as e:
            print('User stopped')
            for i in range(current_task_id + 1, len(tasks)):
                results.append(
                    {'im': None, 'seed': task['task_seed'], 'finish_reason': GenerationFinishReason.user_cancel})
            break
        except Exception as e:
            print('Process failed:', e)
            process_with_error = True
            results.append(
                {'im': None, 'seed': task['task_seed'], 'finish_reason': GenerationFinishReason.error})
            
    print(f"[Task Queue] Finish task, seq={task_seq}")
    task_queue.finish_task(task_seq, results, process_with_error)

    if streaming_output:
        bytes = narray_to_bytesimg(results[0]['im'])
        return Response(bytes, media_type='image/png')
    else:
        results = [GeneratedImageItem(base64=narray_to_base64img(
            item['im']), seed=item['seed'], finish_reason=item['finish_reason']) for item in results]
        return results


def start_app(args):
    uvicorn.run("fooocusapi.api:app", host=args.host,
                port=args.port, log_level=args.log_level)
