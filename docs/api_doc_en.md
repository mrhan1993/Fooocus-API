- [Introduction](#introduction)
- [Fooocus capability related interfaces](#fooocus-capability-related-interfaces)
  - [text-to-image](#text-to-image)
  - [image-upscale-vary](#image-upscale-vary)
  - [image-inpaint-outpaint](#image-inpaint-outpaint)
  - [image-prompt](#image-prompt)
  - [text-to-image-with-imageprompt](#text-to-image-with-imageprompt)
  - [describe](#describe)
  - [all-models](#all-models)
  - [refresh-models](#refresh-models)
  - [styles](#styles)
- [Fooocus API task related interfaces](#fooocus-api-task-related-interfaces)
  - [job-queue](#job-queue)
  - [query-job](#query-job)
  - [job-history](#job-history)
  - [stop](#stop)
  - [ping](#ping)
- [webhook](#webhook)
- [public requests body](#public-requests-params)
  - [AdvanceParams](#advanceparams)
  - [lora](#lora)
  - [response](#response)



# Introduction

Fooocus API are provided more than a dozen REST interfaces now, I roughly divide it into two categories, the first is the ability to call Fooocus, such as generating images, refreshing models, and so on, and the second is related to Fooocus API itself, mainly related to task queries. I will try to illustrate their role and usage and provide examples in the following content.

> Almost all interface parameters have default values, which means you only need to send the parameters you are interested in. The complete parameters and default values can be viewed in the table.

# Fooocus capability related interfaces

## text-to-image

Corresponding to the function of text to image in Fooocus

**base info：**

```yaml
EndPoint: /v1/generation/text-to-image
Method: Post
DataType: json
```
**requests params：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| prompt | string | prompt, default to empty string |
| negative_prompt | string | negative_prompt |
| style | List[str] | list of style, must be supported style, you can get all supported [style](#styles) here |
| performance_selection | Enum | performance_selection, must be one of `Speed`, `Quality`, `Extreme Speed` default to `Speed`|
| aspect_ratios_selection | str | resolution, default to  `1152*896` |
| image_number | int | the num of image to generate, default to 1 , max num is 32, note: Not a parallel interface |
| image_seed | int | seed, default to -1, meant random |
| sharpness | float | sharpness, default to 2.0 , 0-30 |
| guidance_scale | float | guidance scale, default to 4.0 , 1-30 |
| base_model_name | str | base model, default to `juggernautXL_version6Rundiffusion.safetensors` |
| refiner_model_name | str | refiner model, default to `None` |
| refiner_switch | float | refiner switch, default to 0.5 |
| loras | List[Lora] | lora list, include conf, lora: [Lora](#lora) |
| advanced_params | AdvacedParams | Adavanced params, [AdvancedParams](#advanceparams) |
| require_base64 | bool | require base64, default to False |
| async_process | bool | is async, default to False |
| webhook_url | str | after async task completed, address for callback, default to None, refer to [webhook](#webhook) |

**response params：**

Most response have the same structure, but different parts will be specifically explained 

This interface returns a universal response structure, refer to [response](#response)

**request example：**

```python
host = "http://127.0.0.1:8888"

def text2img(params: dict) -> dict:
    """
    text to image
    """
    result = requests.post(url=f"{host}/v1/generation/text-to-image",
                           data=json.dumps(params),
                           headers={"Content-Type": "application/json"})
    return result.json()

result =text2img({
    "prompt": "1girl sitting on the ground",
    "async_process": True})
print(result)
```

## image-upscale-vary

Corresponding to the function of Upscale or Variation in Fooocus

the requests body for this interface based on [text-to-image](#text-to-image), so i will only list the difference with [text-to-image](#text-to-image)

In addition, the interface provides two versions, and there is no functional difference between the two versions, mainly due to slight differences in request methods

**base info：**

```yaml
EndPoint_V1: /v1/generation/image-upscale-vary
EndPoint_V2: /v2/generation/image-upscale-vary
Method: Post
DataType: form|json
```

### V1

**requests params**

| Name | Type | Description             |
| ---- | ---- |---------------------------|
| input_image | string($binary) | binary imagge                                                                                                                             |
| uov_method | Enum | 'Vary (Subtle)','Vary (Strong)','Upscale (1.5x)','Upscale (2x)','Upscale (Fast 2x)','Upscale (Custom)'                                    |
| upscale_value | float | default to None , 1.0-5.0, magnification, only for uov_method is 'Upscale (Custom)'                                                       |
| style | List[str] | list Fooocus style seg with comma                                                                                                         |
| loras | str(List[Lora]) | list for lora, with configure, lora: [Lora](#lora), example: [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}] |
| advanced_params | str(AdvacedParams) | AdvancedParams, AdvancedParams: [AdvancedParams](#advanceparams), send with str, None is available                                        |

**response params：**

This interface returns a universal response structure, refer to [response](#response)

**requests example：**

```python
# headers should not contain {"Content-Type": "application/json"}

host = "http://127.0.0.1:8888"
image = open("./examples/imgs/bear.jpg", "rb").read()

def upscale_vary(image, params: dict) -> dict:
    """
    Upscale or Vary
    """
    response = requests.post(url=f"{host}/v1/generation/image-upscale-vary",
                        data=params,
                        files={"input_image": image})
    return response.json()

result =upscale_vary(image=image,
                     params={
                         "uov_method": "Upscale (2x)",
                         "async_process": True
                     })
print(json.dumps(result, indent=4, ensure_ascii=False))
```

### V2

**requests params**

| Name | Type | Description                                                                                                                           |
| ---- | ---- |---------------------------------------------------------------------------------------------------------------------------------------|
| uov_method | UpscaleOrVaryMethod | Enum type, value should one of 'Vary (Subtle)','Vary (Strong)','Upscale (1.5x)','Upscale (2x)','Upscale (Fast 2x)','Upscale (Custom)' |
| upscale_value | float | default to None , 1.0-5.0, magnification, only for uov_method is 'Upscale (Custom)'                                                   |
| input_image | str | input image, base64 str, or a URL                                                                                                               |

**response params：**

This interface returns a universal response structure, refer to [response](#response)

**requests params：**

```python
host = "http://127.0.0.1:8888"
image = open("./examples/imgs/bear.jpg", "rb").read()

def upscale_vary(image, params: dict) -> dict:
    """
    Upscale or Vary
    """
    params["input_image"] = base64.b64encode(image).decode('utf-8') 
    response = requests.post(url=f"{host}/v2/generation/image-upscale-vary",
                        data=json.dumps(params),
                        headers={"Content-Type": "application/json"},
                        timeout=300)
    return response.json()

result =upscale_vary(image=image,
                     params={
                         "uov_method": "Upscale (2x)",
                         "async_process": True
                     })
print(json.dumps(result, indent=4, ensure_ascii=False))
```

## image-inpaint-outpaint

**base info：**

```yaml
EndPoint_V1: /v1/generation/image-inpait-outpaint
EndPoint_V2: /v2/generation/image-inpait-outpaint
Method: Post
DataType: form|json
```

### V1

**requests params**

| Name | Type | Description                                                                                                               |
| ---- | ---- |---------------------------------------------------------------------------------------------------------------------------|
| input_image | string($binary) | binary imagge                                                                                                             |
| input_mask | string($binary) | binary imagge                                                                                                             |
| inpaint_additional_prompt | string | additional_prompt                                                                                                         |
| outpaint_selections | str | Image extension direction , 'Left', 'Right', 'Top', 'Bottom' seg with comma                                               |
| outpaint_distance_left | int | Image extension distance, default to 0                                                                                    |
| outpaint_distance_right | int | Image extension distance, default to 0                                                                                                             |
| outpaint_distance_top | int | Image extension distance, default to 0                                                                                                             |
| outpaint_distance_bottom | int | Image extension distance, default to 0                                                                                                             |
| style | List[str] | list Fooocus style seg with comma                                                                                                       |
| loras | str(List[Lora]) | list for lora, with configure, lora: Lora, example: [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}] |
| advanced_params | str(AdvacedParams) | AdvancedParams, AdvancedParams: AdvancedParams, send with str, None is available                                                  |

**response params：**

This interface returns a universal response structure, refer to [response](#response)

**requests example：**

```python
# example for inpaint outpaint v1
host = "http://127.0.0.1:8888"
image = open("./examples/imgs/bear.jpg", "rb").read()

def inpaint_outpaint(params: dict, input_image: bytes, input_mask: bytes = None) -> dict:
    """
    example for inpaint outpaint v1
    """
    response = requests.post(url=f"{host}/v1/generation/image-inpait-outpaint",
                        data=params,
                        files={"input_image": input_image,
                               "input_mask": input_mask})
    return response.json()

# image extension example
result = inpaint_outpaint(params={
                            "outpaint_selections": "Left,Right",
                            "async_process": True},
                          input_image=image,
                          input_mask=None)
print(json.dumps(result, indent=4, ensure_ascii=False))

# image inpaint example
source = open("./examples/imgs/s.jpg", "rb").read()
mask = open("./examples/imgs/m.png", "rb").read()
result = inpaint_outpaint(params={
                            "prompt": "a cat",
                            "async_process": True},
                          input_image=source,
                          input_mask=mask)
print(json.dumps(result, indent=4, ensure_ascii=False))
```

### V2

**requests params**

| Name | Type | Description                                                                     |
| ---- | ---- |---------------------------------------------------------------------------------|
| input_image | str | input image, base64 str, or a URL                                                         |
| input_mask | str | input mask, base64 str, or a URL                                                          |
| inpaint_additional_prompt | str | additional prompt                                                               |
| outpaint_selections | List[OutpaintExpansion] | OutpaintExpansion is Enum, value shoule one of "Left", "Right", "Top", "Bottom" |
| outpaint_distance_left | int | Image extension distance, default to 0                                                                    |
| outpaint_distance_right | int | Image extension distance, default to 0                                                                    |
| outpaint_distance_top | int | Image extension distance, default to 0                                                                    |
| outpaint_distance_bottom | int | Image extension distance, default to 0                                                                    |

**response params：**

This interface returns a universal response structure, refer to [response](#response)[response params](#response)

**requests example：**

```python
# example for inpaint outpaint v2
host = "http://127.0.0.1:8888"
image = open("./examples/imgs/bear.jpg", "rb").read()

def inpaint_outpaint(params: dict) -> dict:
    """
    example for inpaint outpaint v2
    """
    response = requests.post(url=f"{host}/v2/generation/image-inpait-outpaint",
                        data=json.dumps(params),
                        headers={"Content-Type": "application/json"})
    return response.json()

# image extension example
result = inpaint_outpaint(params={
                            "input_image": base64.b64encode(image).decode('utf-8'),
                            "input_mask": None,
                            "outpaint_selections": ["Left", "Right"],
                            "async_process": True})
print(json.dumps(result, indent=4, ensure_ascii=False))

# image inpaint example
source = open("./examples/imgs/s.jpg", "rb").read()
mask = open("./examples/imgs/m.png", "rb").read()
result = inpaint_outpaint(params={
                            "prompt": "a cat",
                            "input_image": base64.b64encode(source).decode('utf-8'),
                            "input_mask": base64.b64encode(mask).decode('utf-8'),
                            "async_process": True})
print(json.dumps(result, indent=4, ensure_ascii=False))
```

## image-prompt

`v0.3.27` has a break change. Interface based on change to [inpaint-outpaint](#image-inpaint-outpaint)

after v0.3.27, this interface implements the functions of `inpaint_outpaint` and `image-prompt`.

>  Multi-function interface, which does not implement the functions of `inpaint_outpaint` and `image-prompt` at the same time in the same request

**base info：**

```yaml
EndPoint_V1: /v1/generation/image-prompt
EndPoint_V2: /v2/generation/image-prompt
Method: Post
DataType: form|json
```

### V1

**requests params**

| Name | Type | Description                    |
| ---- | ---- |--------------------------------|
| input_image | Bytes | binary image, use for inpaint    |
| input_mask | Bytes | binary image mask, use for inpaint |
| inpaint_additional_prompt | str | inpaint additional prompt |
| outpaint_selections | str | Image extension direction , 'Left', 'Right', 'Top', 'Bottom' seg with comma |
| outpaint_distance_left | int | Image extension distance, default to 0 |
| outpaint_distance_right | int | Image extension distance, default to 0 |
| outpaint_distance_top | int | Image extension distance, default to 0 |
| outpaint_distance_bottom | int | Image extension distance, default to 0 |
| cn_img1 | string($binary) | binary image                                                                                                                    |
| cn_stop1 | float | default to 0.6                                                                                                                  |
| cn_weight1 | float | default to 0.6                                                                                                                  |
| cn_type1 | Emum | should one of "ImagePrompt", "FaceSwap", "PyraCanny", "CPDS"                                                                             |
| cn_img2 | string($binary) | binary image                                                                                                                    |
| cn_stop2 | float | default to 0.6                                                                                                                  |
| cn_weight2 | float | default to 0.6                                                                                                                  |
| cn_type2 | Emum | should one of "ImagePrompt", "FaceSwap", "PyraCanny", "CPDS"                                                                             |
| cn_img3 | string($binary) | binary image                                                                                                                    |
| cn_stop3 | float | default to 0.6                                                                                                                  |
| cn_weight3 | float | default to 0.6                                                                                                                  |
| cn_type3 | Emum | should one of "ImagePrompt", "FaceSwap", "PyraCanny", "CPDS"                                                                             |
| cn_img4 | string($binary) | binary image                                                                                                                    |
| cn_stop4 | float | default to 0.6                                                                                                                  |
| cn_weight4 | float | default to 0.6                                                                                                                  |
| cn_type4 | Emum | should one of "ImagePrompt", "FaceSwap", "PyraCanny", "CPDS"                                                                |
| style | List[str] | list Fooocus style seg with comma                                                                                               |
| loras | str(List[Lora]) | list for lora, with configure, lora: Lora, example: [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}] |
| advanced_params | str(AdvacedParams) | AdvancedParams, AdvancedParams: AdvancedParams, send with str, None is available                                                |

**response params：**

This interface returns a universal response structure, refer to [response](#response)[response params](#response)

**requests example：**

```python
# image_prompt v1 example
host = "http://127.0.0.1:8888"
image = open("./examples/imgs/bear.jpg", "rb").read()
source = open("./examples/imgs/s.jpg", "rb").read()
mask = open("./examples/imgs/m.png", "rb").read()

def image_prompt(params: dict,
                 input_iamge: bytes=None,
                 input_mask: bytes=None,
                 cn_img1: bytes=None,
                 cn_img2: bytes=None,
                 cn_img3: bytes=None,
                 cn_img4: bytes=None,) -> dict:
    """
    image prompt
    """
    response = requests.post(url=f"{host}/v1/generation/image-prompt",
                             data=params,
                             files={
                                 "input_image": input_iamge,
                                 "input_mask": input_mask,
                                 "cn_img1": cn_img1,
                                 "cn_img2": cn_img2,
                                 "cn_img3": cn_img3,
                                 "cn_img4": cn_img4,
                              })
    return response.json()

# image extend
params = {
    "outpaint_selections": ["Left", "Right"],
    "image_prompts": [] # 必传参数，可以为空列表
}
result = image_prompt(params=params, input_iamge=image)
print(json.dumps(result, indent=4, ensure_ascii=False))

# inpaint

params = {
    "prompt": "1girl sitting on the chair",
    "image_prompts": [], # required, can be empty list
    "async_process": True
}
result = image_prompt(params=params, input_iamge=source, input_mask=mask)
print(json.dumps(result, indent=4, ensure_ascii=False))

# image prompt

params = {
    "prompt": "1girl sitting on the chair",
    "image_prompts": [
        {
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        },{
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        }]
    }
result = image_prompt(params=params, cn_img1=image, cn_img2=source)
print(json.dumps(result, indent=4, ensure_ascii=False))
```

### V2

**requests params**

| Name | Type | Description                                     |
| ---- | ---- |-------------------------------------------------|
| input_image | str | base64 image, or a URL, use for inpaint    |
| input_mask | str | base64 image mask, or a URL, use for inpaint |
| inpaint_additional_prompt | str | inpaint additional prompt |
| outpaint_selections | List[] | Image extension direction , 'Left', 'Right', 'Top', 'Bottom' seg with comma |
| outpaint_distance_left | int | Image extension distance, default to 0 |
| outpaint_distance_right | int | Image extension distance, default to 0 |
| outpaint_distance_top | int | Image extension distance, default to 0 |
| outpaint_distance_bottom | int | Image extension distance, default to 0 |
| image_prompts | List[ImagePrompt] | image list, include config, ImagePrompt struct： |

**ImagePrompt**

| Name | Type | Description                                                                         |
| ---- | ---- |-------------------------------------------------------------------------------------|
| cn_img | str | input image, base64 str, or a URL                                                             |
| cn_stop | float | 0-1, default to 0.5                                                                 |
| cn_weight | float | weight, 0-2, default to 1.0                                                         |
| cn_type | ControlNetType | ControlNetType Enum, should one of "ImagePrompt", "FaceSwap", "PyraCanny", "CPDS" |

**response params：**

This interface returns a universal response structure, refer to [response](#response)[response params](#response)

**requests example：**

```python
# image_prompt v2 example
host = "http://127.0.0.1:8888"
image = open("./examples/imgs/bear.jpg", "rb").read()
source = open("./examples/imgs/s.jpg", "rb").read()
mask = open("./examples/imgs/m.png", "rb").read()

def image_prompt(params: dict) -> dict:
    """
    image prompt
    """
    response = requests.post(url=f"{host}/v2/generation/image-prompt",
                             data=json.dumps(params),
                             headers={"Content-Type": "application/json"})
    return response.json()

# image extend
params = {
    "input_image": base64.b64encode(image).decode('utf-8'),
    "outpaint_selections": ["Left", "Right"],
    "image_prompts": [] # required, can be empty list
}
result = image_prompt(params)
print(json.dumps(result, indent=4, ensure_ascii=False))

# inpaint

params = {
    "prompt": "1girl sitting on the chair",
    "input_image": base64.b64encode(source).decode('utf-8'),
    "input_mask": base64.b64encode(mask).decode('utf-8'),
    "image_prompts": [], # required, can be empty list
    "async_process": True
}
result = image_prompt(params)
print(json.dumps(result, indent=4, ensure_ascii=False))

# image prompt

params = {
    "prompt": "1girl sitting on the chair",
    "image_prompts": [
        {
            "cn_img": base64.b64encode(source).decode('utf-8'),
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        },{
            "cn_img": base64.b64encode(image).decode('utf-8'),
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        }]
    }
result = image_prompt(params)
print(json.dumps(result, indent=4, ensure_ascii=False))
```

## text to image with imageprompt

this interface only provides v2 version

**base info:**

```yaml
EndPoint: /v2/generation/text-to-image-with-ip
Method: Post
DataType: json
```

**requests params**

| Name | Type | Description |
| ---- | ---- | ----------- |
| image_prompts | List[ImagePrompt] | 图像列表 |

**requests example**:

```python
# text to image with imageprompt example
host = "http://127.0.0.1:8888"
image = open("./examples/imgs/bear.jpg", "rb").read()
source = open("./examples/imgs/s.jpg", "rb").read()
def image_prompt(params: dict) -> dict:
    """
    image prompt
    """
    response = requests.post(url=f"{host}/v2/generation/text-to-image-with-ip",
                             data=json.dumps(params),
                             headers={"Content-Type": "application/json"})
    return response.json()

params = {
    "prompt": "A bear",
    "image_prompts": [
        {
            "cn_img": base64.b64encode(source).decode('utf-8'),
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        },{
            "cn_img": base64.b64encode(image).decode('utf-8'),
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        }
    ]
}
result = image_prompt(params)
print(json.dumps(result, indent=4, ensure_ascii=False))
```

## describe

**base info：**

```yaml
EndPoint: /v1/tools/describe-image
Method: Post
DataType: form
```

**requests params**

| Name | Type | Description                              |
|------|------|------------------------------------------|
| type | Enum | type, should be one of "Photo", "Anime"  |

**requests example**:

```python
def describe_image(image: bytes,
                   params: dict = {"type": "Photo"}) -> dict:
    """
    describe-image
    """
    response = requests.post(url="http://127.0.0.1:8888/v1/tools/describe-image",
                        files={
                            "image": image
                        },
                        timeout=30)
    return response.json()
```

**response example**:

```python
{
  "describe": "a young woman posing with her hands behind her head"
}
```

--------------------------------------------

## all-models

**base info：**

```yaml
EndPoint: /v1/engines/all-models
Method: Get
```

**requests example**:

```python
def all_models() -> dict:
    """
    all-models
    """
    response = requests.get(url="http://127.0.0.1:8888/v1/engines/all-models",
                        timeout=30)
    return response.json()
```

**response params**:

```python
{
  "model_filenames": [
    "juggernautXL_version6Rundiffusion.safetensors",
    "sd_xl_base_1.0_0.9vae.safetensors",
    "sd_xl_refiner_1.0_0.9vae.safetensors"
  ],
  "lora_filenames": [
    "sd_xl_offset_example-lora_1.0.safetensors"
  ]
}
```

## refresh-models

**base info：**

```yaml
EndPoint: /v1/engines/refresh-models
Method: Post
```

**requests example**
```python
def refresh() -> dict:
    """
    refresh-models
    """
    response = requests.post(url="http://127.0.0.1:8888/v1/engines/refresh-models",
                        timeout=30)
    return response.json()
```

**response params**
```python
{
  "model_filenames": [
    "juggernautXL_version6Rundiffusion.safetensors",
    "sd_xl_base_1.0_0.9vae.safetensors",
    "sd_xl_refiner_1.0_0.9vae.safetensors"
  ],
  "lora_filenames": [
    "sd_xl_offset_example-lora_1.0.safetensors"
  ]
}
```

## styles

**base info：**

```yaml
EndPoint: /v1/engines/styles
Method: Get
```

**requests example**:

```python
def styles() -> dict:
    """
    styles
    """
    response = requests.get(url="http://127.0.0.1:8888/v1/engines/styles",
                        timeout=30)
    return response.json()
```

**response params**:

```python
[
  "Fooocus V2",
  "Fooocus Enhance",
  ...
  "Watercolor 2",
  "Whimsical And Playful"
]
```

# Fooocus API task related interfaces

## job-queue

**base info：**

```yaml
EndPoint: /v1/engines/job-queue
Method: Get
```

**requests example**:

```python
def job_queue() -> dict:
    """
    job-queue
    """
    response = requests.get(url="http://127.0.0.1:8888/v1/generation/job-queue",
                        timeout=30)
    return response.json()
```

**response params**:

```python
{
  "running_size": 0,
  "finished_size": 1,
  "last_job_id": "cac3914a-926d-4b6f-a46a-83794a0ce1d4"
}
```

## query-job

**base info：**

```yaml
EndPoint: /v1/generation/query-job
Method: Get
```

**requests example**:
```python
def taskResult(task_id: str) -> dict:
    # get task status
    task_status = requests.get(url="http://127.0.0.1:8888/v1/generation/query-job",
                               params={"job_id": task_id,
                                       "require_step_preivew": False},
                               timeout=30)

    return task_status.json()
```

**response params**:
```python
{
  "job_id": "cac3914a-926d-4b6f-a46a-83794a0ce1d4",
  "job_type": "Text to Image",
  "job_stage": "SUCCESS",
  "job_progress": 100,
  "job_status": "Finished",
  "job_step_preview": null,
  "job_result": [
    {
      "base64": null,
      "url": "http://127.0.0.1:8888/files/2023-11-27/b928e50e-3c09-4187-a3f9-1c12280bfd95.png",
      "seed": 8228839561385006000,
      "finish_reason": "SUCCESS"
    }
  ]
}
```

## job-history

**base info：**

```yaml
EndPoint: /v1/generation/job-history
Method: get
```

**requests example**:

```python
def job-history() -> dict:
    """
    job-history
    """
    response = requests.get(url="http://127.0.0.1:8888/v1/generation/job-history",
                        timeout=30)
    return response.json()
```

**response params**:

```python
{
  "queue": [],
  "history": [
    "job_id": "cac3914a-926d-4b6f-a46a-83794a0ce1d4",
    "is_finished": True
  ]
}
```

## stop

**base info：**

```yaml
EndPoint: /v1/generation/stop
Method: post
```

**requests example**:

```python
def stop() -> dict:
    """
    stop
    """
    response = requests.post(url="http://127.0.0.1:8888/v1/generation/stop",
                        timeout=30)
    return response.json()
```

**response params**:

```python
{
  "msg": "success"
}
```

## ping

**base info：**

```yaml
EndPoint: /ping
Method: get
```

pong

# webhook

You can specify an address through '--webhook_url' on the command line so that you can receive notifications after asynchronous tasks are completed

Here is a simple example to demonstrate how 'webhook' works

First，start a simple server using the following code:

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/status")
async def status(requests: dict):
    print(requests)

uvicorn.run(app, host="0.0.0.0", port=8000)
```

Then, start Fooocus API with `--webhook-url http://host:8000/status`

Submit a task in any way, and after completion, you will see the task completion information in the background of this simple server：

```python
{'job_id': '717ec0b5-85df-4174-80d6-bddf93cd8248', 'job_result': [{'url': 'http://127.0.0.1:8888/files/2023-12-29/f1eca704-718e-4781-9d5f-82d41aa799d7.png', 'seed': '3283449865282320931'}]}
```

# public requests params

## AdvanceParams

| Name | Type | Description                                                                      |
| ---- | ---- |----------------------------------------------------------------------------------|
| disable_preview | bool | disable preview, default to False                                                |
| adm_scaler_positive | float | ADM Guidance Scaler, default to 1.5, range 0.1-3.0                               |
| adm_scaler_negative | float | negative ADM Guidance Scaler, default to 0.8, range 0.1-3.0                      |
| adm_scaler_end | float | ADM Guidance Scaler end value, default to 0.5, range 0.0-1.0                     |
| refiner_swap_method | str | refiner model swap method, default to `joint`                                    |
| adaptive_cfg | float | CFG Mimicking from TSNR, default to 7.0, range 1.0-30.0                          |
| sampler_name | str | sampler, default to `default_sampler`                                            |
| scheduler_name | str | scheduler, default to `default_scheduler`                                        |
| overwrite_step | int | Forced Overwrite of Sampling Step, default to -1, range -1-200                   |
| overwrite_switch | int | Forced Overwrite of Refiner Switch Step, default to -1, range -1-200             |
| overwrite_width | int | Forced Overwrite of Generating Width, default to -1, range -1-2048               |
| overwrite_height | int | Forced Overwrite of Generating Height, default to -1, range -1-2048              |
| overwrite_vary_strength | float | Forced Overwrite of Denoising Strength of "Vary", default to -1, range -1-1.0    |
| overwrite_upscale_strength | float | Forced Overwrite of Denoising Strength of "Upscale", default to -1, range -1-1.0 |
| mixing_image_prompt_and_vary_upscale | bool | Mixing Image Prompt and Vary/Upscale, default to False                           |
| mixing_image_prompt_and_inpaint | bool | Mixing Image Prompt and Inpaint, default to False                                |
| debugging_cn_preprocessor | bool | Debug Preprocessors, default to False                                            |
| skipping_cn_preprocessor | bool | Skip Preprocessors, default to False                                             |
| controlnet_softness | float | Softness of ControlNet, default to 0.25, range 0.0-1.0                           |
| canny_low_threshold | int | Canny Low Threshold, default to 64, range 1-255                                  |
| canny_high_threshold | int | Canny High Threshold, default to 128, range 1-255                                |
| freeu_enabled | bool | FreeU enabled, default to False                                                  |
| freeu_b1 | float | FreeU B1, default to 1.01                                                        |
| freeu_b2 | float | FreeU B2, default to 1.02                                                        |
| freeu_s1 | float | FreeU B3, default to 0.99                                                        |
| freeu_s2 | float | FreeU B4, default to 0.95                                                        |
| debugging_inpaint_preprocessor | bool | Debug Inpaint Preprocessing, default to False                                    |
| inpaint_disable_initial_latent | bool | Disable initial latent in inpaint, default to False                              |
| inpaint_engine | str | Inpaint Engine, default to `v1`                                                  |
| inpaint_strength | float | Inpaint Denoising Strength, default to 1.0, range 0.0-1.0                        |
| inpaint_respective_field | float | Inpaint Respective Field, default to 1.0, range 0.0-1.0                          |

## lora

| Name | Type | Description            |
| ---- | ---- |------------------------|
| model_name | str | model name             |
| weight | float | weight, default to 0.5 |

## response

success response：

**async_process: True**

| Name | Type | Description  |
| ---- | ---- |--------------|
| job_id | int | job ID       |
| job_type | str | job type     |
| job_stage | str | job stage    |
| job_progress | float | job progress |
| job_status | str | job status   |
| job_step_preview | str | job previes  |
| job_result | str | job result   |

**async_process: False**

| Name | Type | Description                                                                      |
| ---- | ---- |----------------------------------------------------------------------------------|
| base64 | str | base64 image, according to `require_base64` params determines whether it is null |
| url | str | result image url                                                                 |
| seed | int | image seed                                                                       |
| finish_reason | str | finish reason                                                                    |

fail response：
