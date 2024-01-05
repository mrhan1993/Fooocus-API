- [简介](#简介)
- [Fooocus 能力相关接口](#fooocus-能力相关接口)
  - [文生图 | text-to-image](#文生图--text-to-image)
  - [图像放大 | image-upscale-vary](#图像放大--image-upscale-vary)
  - [局部重绘 | image-inpaint-outpaint](#局部重绘--image-inpaint-outpaint)
  - [图生图 | image-prompt](#图生图--image-prompt)
  - [text-to-image-with-imageprompt](#text-to-image-with-imageprompt)
  - [图像反推 | describe](#图像反推--describe)
  - [列出模型 | all-models](#列出模型--all-models)
  - [刷新模型 | refresh-models](#刷新模型--refresh-models)
  - [样式 | styles](#样式--styles)
- [Fooocus API 任务相关接口](#fooocus-api-任务相关接口)
  - [任务队列 | job-queue](#任务队列--job-queue)
  - [查询任务 | query-job](#查询任务--query-job)
  - [查询任务历史 | job-history](#查询任务历史--job-history)
  - [停止任务 | stop](#停止任务--stop)
  - [ping](#ping)
- [webhook](#webhook)
- [公共请求体](#公共请求体)
  - [高级参数 | AdvanceParams](#高级参数--advanceparams)
  - [lora](#lora)
  - [响应参数 | response](#响应参数--response)



# 简介

Fooocus API 目前提供了十多个 REST 接口, 我大致将其分为两类, 第一类用来调用 Fooocus 的能力, 比如生成图像、刷新模型之类的, 第二类为 Fooocus API 自身相关的, 主要是任务查询相关。我会在接下来的内容中尝试说明它们的作用以及用法并提供示例。

> 几乎所有的接口参数都有默认值，这意味着你只需要发送你感兴趣的参数即可。完整的参数以及默认值可以通过表格查看

# Fooocus 能力相关接口

## 文生图 | text-to-image

对应 Fooocus 中的文生图功能

**基础信息：**

```yaml
EndPoint: /v1/generation/text-to-image
Method: Post
DataType: json
```
**请求参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| prompt | string | 描述词, 默认为空字符串 |
| negative_prompt | string | 描述词, 反向描述词 |
| style | List[str] | 风格列表, 需要是受支持的风格, 可以通过 [样式接口](#样式--styles) 获取所有支持的样式 |
| performance_selection | Enum | 性能选择, `Speed`, `Quality`, `Extreme Speed` 中的一个, 默认 `Speed`|
| aspect_ratios_selection | str | 分辨率, 默认 '1152*896' |
| image_number | int | 生成图片数量, 默认 1 , 最大32, 注: 非并行接口 |
| image_seed | int | 图片种子, 默认 -1, 即随机生成 |
| sharpness | float | 锐度, 默认 2.0 , 0-30 |
| guidance_scale | float | 引导比例, 默认 4.0 , 1-30 |
| base_model_name | str | 基础模型, 默认 `juggernautXL_version6Rundiffusion.safetensors` |
| refiner_model_name | str | 优化模型, 默认 `None` |
| refiner_switch | float | 优化模型切换时机, 默认 0.5 |
| loras | List[Lora] | lora 模型列表, 包含配置, lora 结构: [Lora](#lora) |
| advanced_params | AdvacedParams | 高级参数, AdvancedParams 结构 [AdvancedParams](#高级参数--advanceparams) |
| require_base64 | bool | 是否返回base64编码, 默认 False |
| async_process | bool | 是否异步处理, 默认 False |
| webhook_url | str | 异步处理完成后, 触发的 webhook 地址, 参考[webhook](#webhook) |

**响应参数：**

多数响应结构式相同的, 不同的部分会进行特别说明.

该接口返回通用响应结构, 参考[响应参数](#响应参数--response)

**请求示例：**

```python
host = "http://127.0.0.1:8888"

def text2img(params: dict) -> dict:
    """
    文生图
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

## 图像放大 | image-upscale-vary

该接口对应 Fooocus 中的 Upscale or Variation 功能

该接口参数继承自[文生图](#文生图--text-to-image), 因此后面只会列出和[文生图](#文生图--text-to-image)请求参数差异部分

此外, 该接口提供了两个版本, 两个版本并无功能上的差异, 主要是请求方式略有区别

**基础信息：**

```yaml
EndPoint_V1: /v1/generation/image-upscale-vary
EndPoint_V2: /v2/generation/image-upscale-vary
Method: Post
DataType: form|json
```

### V1

**请求参数**

| Name | Type | Description |
| ---- | ---- | ----------- |
| input_image | string($binary) | 二进制 str 图像 |
| uov_method | Enum | 'Vary (Subtle)','Vary (Strong)','Upscale (1.5x)','Upscale (2x)','Upscale (Fast 2x)','Upscale (Custom)' |
| upscale_value | float | 默认为 None , 1.0-5.0, 放大倍数, 仅在 'Upscale (Custom)' 中有效 |
| style | List[str] | 以逗号分割的 Fooocus 风格列表 |
| loras | str(List[Lora]) | lora 模型列表, 包含配置, lora 结构: [Lora](#lora), 比如: [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}] |
| advanced_params | str(AdvacedParams) | 高级参数, AdvancedParams 结构 [AdvancedParams](#高级参数--advanceparams), 以字符串形式发送, 可以为空 |

**响应参数：**

该接口返回通用响应结构, 参考[响应参数](#响应参数--response)

**请求示例：**

```python
# 不要加 {"Content-Type": "application/json"} 这个 header

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

**请求参数**

| Name | Type | Description |
| ---- | ---- | ----------- |
| uov_method | UpscaleOrVaryMethod | 是个枚举类型, 包括 'Vary (Subtle)','Vary (Strong)','Upscale (1.5x)','Upscale (2x)','Upscale (Fast 2x)','Upscale (Custom)' |
| upscale_value | float | 默认为 None , 1.0-5.0, 放大倍数, 仅在 'Upscale (Custom)' 中有效 |
| input_image | str | 输入图像, base64 格式, 或者一个URL |

**响应参数：**

该接口返回通用响应结构, 参考[响应参数](#响应参数--response)

**请求示例：**

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

## 局部重绘 | image-inpaint-outpaint

**基础信息：**

```yaml
EndPoint_V1: /v1/generation/image-inpait-outpaint
EndPoint_V2: /v2/generation/image-inpait-outpaint
Method: Post
DataType: form|json
```

### V1

**请求参数**

| Name | Type | Description |
| ---- | ---- | ----------- |
| input_image | string($binary) | 二进制 str 图像 |
| input_mask | string($binary) | 二进制 str 图像 |
| inpaint_additional_prompt | string | 附加描述 |
| outpaint_selections | str | 图像扩展方向, 逗号分割的 'Left', 'Right', 'Top', 'Bottom' |
| outpaint_distance_left | int | 图像扩展距离, 默认 0 |
| outpaint_distance_right | int | 图像扩展距离, 默认 0 |
| outpaint_distance_top | int | 图像扩展距离, 默认 0 |
| outpaint_distance_bottom | int | 图像扩展距离, 默认 0 |
| style | List[str] | 以逗号分割的 Fooocus 风格列表 |
| loras | str(List[Lora]) | lora 模型列表, 包含配置, lora 结构: [Lora](#lora), 比如: [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}] |
| advanced_params | str(AdvacedParams) | 高级参数, AdvancedParams 结构 [AdvancedParams](#高级参数--advanceparams), 以字符串形式发送 |

**响应参数：**

该接口返回通用响应结构, 参考[响应参数](#响应参数--response)

**请求示例：**

```python
# 局部重绘 v1 接口示例
host = "http://127.0.0.1:8888"
image = open("./examples/imgs/bear.jpg", "rb").read()

def inpaint_outpaint(params: dict, input_image: bytes, input_mask: bytes = None) -> dict:
    """
    局部重绘 v1 接口示例
    """
    response = requests.post(url=f"{host}/v1/generation/image-inpait-outpaint",
                        data=params,
                        files={"input_image": input_image,
                               "input_mask": input_mask})
    return response.json()

# 图片扩展示例
result = inpaint_outpaint(params={
                            "outpaint_selections": "Left,Right",
                            "async_process": True},
                          input_image=image,
                          input_mask=None)
print(json.dumps(result, indent=4, ensure_ascii=False))

# 局部重绘示例
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

**请求参数**

| Name | Type | Description                                                     |
| ---- | ---- |-----------------------------------------------------------------|
| input_image | str | 输入图像, base64 格式, 或者一个URL                                                 |
| input_mask | str | 输入遮罩, base64 格式, 或者一个URL                                                 |
| inpaint_additional_prompt | str | 附加描述词                                                           |
| outpaint_selections | List[OutpaintExpansion] | OutpaintExpansion 是一个枚举类型, 值包括 "Left", "Right", "Top", "Bottom" |
| outpaint_distance_left | int | 图像扩展距离, 默认 0                                                    |
| outpaint_distance_right | int | 图像扩展距离, 默认 0                                                    |
| outpaint_distance_top | int | 图像扩展距离, 默认 0                                                    |
| outpaint_distance_bottom | int | 图像扩展距离, 默认 0                                                    |

**响应参数：**

该接口返回通用响应结构, 参考[响应参数](#响应参数--response)

**请求示例：**

```python
# 局部重绘 v2 接口示例
host = "http://127.0.0.1:8888"
image = open("./examples/imgs/bear.jpg", "rb").read()

def inpaint_outpaint(params: dict) -> dict:
    """
    局部重绘 v1 接口示例
    """
    response = requests.post(url=f"{host}/v2/generation/image-inpait-outpaint",
                        data=json.dumps(params),
                        headers={"Content-Type": "application/json"})
    return response.json()

# 图像扩展示例
result = inpaint_outpaint(params={
                            "input_image": base64.b64encode(image).decode('utf-8'),
                            "input_mask": None,
                            "outpaint_selections": ["Left", "Right"],
                            "async_process": True})
print(json.dumps(result, indent=4, ensure_ascii=False))

# 局部重绘示例
source = open("./examples/imgs/s.jpg", "rb").read()
mask = open("./examples/imgs/m.png", "rb").read()
result = inpaint_outpaint(params={
                            "prompt": "a cat",
                            "input_image": base64.b64encode(source).decode('utf-8'),
                            "input_mask": base64.b64encode(mask).decode('utf-8'),
                            "async_process": True})
print(json.dumps(result, indent=4, ensure_ascii=False))
```

## 图生图 | image-prompt

该接口更新自 `v0.3.27` 后有重大更新。从继承自 [文生图](#文生图--text-to-image) 更改为继承自 [局部重绘](#局部重绘--image-inpaint-outpaint)

该版本之后可以通过该接口实现 `inpaint_outpaint` 以及 `image-prompt` 接口的功能

> 多功能接口，并非可以同时实现 `inpaint_outpaint` 以及 `image-prompt` 接口的功能

**基础信息：**

```yaml
EndPoint_V1: /v1/generation/image-prompt
EndPoint_V2: /v2/generation/image-prompt
Method: Post
DataType: form|json
```

### V1

**请求参数**

> 注意: 虽然接口更改为继承自[局部重绘](#局部重绘--image-inpaint-outpaint), 但下方表格展示的仍然继承自[文生图](#文生图--text-to-image), 但参数是完整的

| Name | Type | Description |
| ---- | ---- | ----------- |
| input_image | Bytes | 二进制图像, 用于局部重绘 |
| input_mask | Bytes | 二进制图像遮罩, 用于局部重绘 |
| inpaint_additional_prompt | str | inpaint 附加提示词 |
| outpaint_selections | str | 图像扩展选项, 逗号分割的 "Left", "Right", "Top", "Bottom" |
| outpaint_distance_left | int | 图像扩展距离, 默认 0 |
| outpaint_distance_right | int | 图像扩展距离, 默认 0 |
| outpaint_distance_top | int | 图像扩展距离, 默认 0 |
| outpaint_distance_bottom | int | 图像扩展距离, 默认 0 |
| cn_img1 | string($binary) | 二进制 str 图像 |
| cn_stop1 | float | 默认 0.6 |
| cn_weight1 | float | 默认 0.6 |
| cn_type1 | Emum | "ImagePrompt", "FaceSwap", "PyraCanny", "CPDS" 中的一个 |
| cn_img2 | string($binary) | 二进制 str 图像 |
| cn_stop2 | float | 默认 0.6 |
| cn_weight2 | float | 默认 0.6 |
| cn_type2 | Emum | "ImagePrompt", "FaceSwap", "PyraCanny", "CPDS" 中的一个 |
| cn_img3 | string($binary) | 二进制 str 图像 |
| cn_stop3 | float | 默认 0.6 |
| cn_weight3 | float | 默认 0.6 |
| cn_type3 | Emum | "ImagePrompt", "FaceSwap", "PyraCanny", "CPDS" 中的一个 |
| cn_img4 | string($binary) | 二进制 str 图像 |
| cn_stop4 | float | 默认 0.6 |
| cn_weight4 | float | 默认 0.6 |
| cn_type4 | Emum | "ImagePrompt", "FaceSwap", "PyraCanny", "CPDS" 中的一个 |
| style | List[str] | 以逗号分割的 Fooocus 风格列表 |
| loras | str(List[Lora]) | lora 模型列表, 包含配置, lora 结构: [Lora](#lora), 比如: [{"model_name": "sd_xl_offset_example-lora_1.0.safetensors", "weight": 0.5}] |
| advanced_params | str(AdvacedParams) | 高级参数, AdvancedParams 结构 [AdvancedParams](#高级参数--advanceparams), 以字符串形式发送 |

**响应参数：**

该接口返回通用响应结构, 参考[响应参数](#响应参数--response)

**请求示例：**

```python
# image_prompt v1 接口示例
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

# 图像扩展
params = {
    "outpaint_selections": ["Left", "Right"],
    "image_prompts": [] # 必传参数，可以为空列表
}
result = image_prompt(params=params, input_iamge=image)
print(json.dumps(result, indent=4, ensure_ascii=False))

# 局部重绘

params = {
    "prompt": "1girl sitting on the chair",
    "image_prompts": [], # 必传参数，可以为空列表
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

**请求参数**

| Name | Type | Description |
| ---- | ---- | ----------- |
| input_image | str | base64 图像, 或者一个URL, 用于局部重绘 |
| input_mask | str | base64 图像遮罩, 或者一个URL, 用于局部重绘 |
| inpaint_additional_prompt | str | inpaint 附加提示词 |
| outpaint_selections | List[OutpaintExpansion] | 图像扩展选项, 逗号分割的 "Left", "Right", "Top", "Bottom" |
| outpaint_distance_left | int | 图像扩展距离, 默认 0 |
| outpaint_distance_right | int | 图像扩展距离, 默认 0 |
| outpaint_distance_top | int | 图像扩展距离, 默认 0 |
| outpaint_distance_bottom | int | 图像扩展距离, 默认 0 |
| image_prompts | List[ImagePrompt] | 图像列表, 包含配置, ImagePrompt 结构如下： |

**ImagePrompt**

| Name | Type | Description |
| ---- | ---- | ----------- |
| cn_img | str | 输入图像, base64 编码, 或者一个URL |
| cn_stop | float | 停止位置, 范围 0-1, 默认 0.5 |
| cn_weight | float | 权重, 范围 0-2, 默认 1.0 |
| cn_type | ControlNetType | 控制网络类型, 是一个枚举类型, 包括: "ImagePrompt", "FaceSwap", "PyraCanny", "CPDS" |

**响应参数：**

该接口返回通用响应结构, 参考[响应参数](#响应参数--response)

**请求示例：**

```python
# image_prompt v2 接口示例
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

# 图像扩展
params = {
    "input_image": base64.b64encode(image).decode('utf-8'),
    "outpaint_selections": ["Left", "Right"],
    "image_prompts": [] # 必传参数，可以为空列表
}
result = image_prompt(params)
print(json.dumps(result, indent=4, ensure_ascii=False))

# 局部重绘

params = {
    "prompt": "1girl sitting on the chair",
    "input_image": base64.b64encode(source).decode('utf-8'),
    "input_mask": base64.b64encode(mask).decode('utf-8'),
    "image_prompts": [], # 必传参数，可以为空列表
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

该接口暂无 v1 版本

**基础信息：**

```yaml
EndPoint: /v2/generation/text-to-image-with-ip
Method: Post
DataType: json
```

**请求参数**

| Name | Type | Description |
| ---- | ---- | ----------- |
| image_prompts | List[ImagePrompt] | 图像列表 |

**请求示例**:

```python
# text to image with imageprompt 示例
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

## 图像反推 | describe

**基础信息：**

```yaml
EndPoint: /v1/tools/describe-image
Method: Post
DataType: form
```

**请求参数**

| Name | Type | Description                 |
|------|------|-----------------------------|
| type | Enum | 反推类型, "Photo", "Anime" 中的一个 |

**请求示例**:

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

**响应示例**:

```python
{
  "describe": "a young woman posing with her hands behind her head"
}
```

--------------------------------------------

## 列出模型 | all-models

**基础信息：**

```yaml
EndPoint: /v1/engines/all-models
Method: Get
```

**请求示例**:

```python
def all_models() -> dict:
    """
    all-models
    """
    response = requests.get(url="http://127.0.0.1:8888/v1/engines/all-models",
                        timeout=30)
    return response.json()
```

**响应示例**:

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

## 刷新模型 | refresh-models

**基础信息：**

```yaml
EndPoint: /v1/engines/refresh-models
Method: Post
```

**请求示例**
```python
def refresh() -> dict:
    """
    refresh-models
    """
    response = requests.post(url="http://127.0.0.1:8888/v1/engines/refresh-models",
                        timeout=30)
    return response.json()
```

**响应示例**
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

## 样式 | styles

**基础信息：**

```yaml
EndPoint: /v1/engines/styles
Method: Get
```

**请求示例**:

```python
def styles() -> dict:
    """
    styles
    """
    response = requests.get(url="http://127.0.0.1:8888/v1/engines/styles",
                        timeout=30)
    return response.json()
```

**响应示例**:

```python
[
  "Fooocus V2",
  "Fooocus Enhance",
  ...
  "Watercolor 2",
  "Whimsical And Playful"
]
```

# Fooocus API 任务相关接口

## 任务队列 | job-queue

**基础信息：**

```yaml
EndPoint: /v1/engines/job-queue
Method: Get
```

**请求示例**:

```python
def job_queue() -> dict:
    """
    job-queue
    """
    response = requests.get(url="http://127.0.0.1:8888/v1/generation/job-queue",
                        timeout=30)
    return response.json()
```

**响应示例**:

```python
{
  "running_size": 0,
  "finished_size": 1,
  "last_job_id": "cac3914a-926d-4b6f-a46a-83794a0ce1d4"
}
```

## 查询任务 | query-job

**基础信息：**

```yaml
EndPoint: /v1/generation/query-job
Method: Get
```

**请求示例**:
```python
def taskResult(task_id: str) -> dict:
    # 获取任务状态
    task_status = requests.get(url="http://127.0.0.1:8888/v1/generation/query-job",
                               params={"job_id": task_id,
                                       "require_step_preivew": False},
                               timeout=30)

    return task_status.json()
```

**响应示例**:
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

## 查询任务历史 | job-history

**基础信息：**

```yaml
EndPoint: /v1/generation/job-history
Method: get
```

**请求示例**:

```python
def job-history() -> dict:
    """
    job-history
    """
    response = requests.get(url="http://127.0.0.1:8888/v1/generation/job-history",
                        timeout=30)
    return response.json()
```

**响应示例**:

```python
{
  "queue": [],
  "history": [
    "job_id": "cac3914a-926d-4b6f-a46a-83794a0ce1d4",
    "is_finished": True
  ]
}
```

## 停止任务 | stop

**基础信息：**

```yaml
EndPoint: /v1/generation/stop
Method: post
```

**请求示例**:

```python
def stop() -> dict:
    """
    stop
    """
    response = requests.post(url="http://127.0.0.1:8888/v1/generation/stop",
                        timeout=30)
    return response.json()
```

**响应示例**:

```python
{
  "msg": "success"
}
```

## ping

**基础信息：**

```yaml
EndPoint: /ping
Method: get
```

pong

# webhook

你可以在命令行通过 `--webhook-url` 指定一个地址，以便异步任务完成之后可以收到通知

下面是一个简单的示例来展示 `webhook` 是如何工作的

首先，使用下面的代码启动一个简易服务器:

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/status")
async def status(requests: dict):
    print(requests)

uvicorn.run(app, host="0.0.0.0", port=8000)
```

然后, 在启动 Fooocus API 时添加 `--webhook-url http://host:8000/status`

通过任意方式提交一个任务, 等完成后你会在这个简易服务器的后台看到任务结束信息：

```python
{'job_id': '717ec0b5-85df-4174-80d6-bddf93cd8248', 'job_result': [{'url': 'http://127.0.0.1:8888/files/2023-12-29/f1eca704-718e-4781-9d5f-82d41aa799d7.png', 'seed': '3283449865282320931'}]}
```

# 公共请求体

## 高级参数 | AdvanceParams

| Name | Type | Description |
| ---- | ---- | ----------- |
| disable_preview | bool | 是否禁用预览, 默认 False |
| adm_scaler_positive | float | 正 ADM Guidance Scaler, 默认 1.5, 范围 0.1-3.0 |
| adm_scaler_negative | float | 负 ADM Guidance Scaler, 默认 0.8, 范围 0.1-3.0 |
| adm_scaler_end | float | ADM Guidance Scaler 结束值, 默认 0.5, 范围 0.0-1.0 |
| refiner_swap_method | str | 优化模型交换方法, 默认 `joint` |
| adaptive_cfg | float | CFG Mimicking from TSNR, 默认 7.0, 范围 1.0-30.0 |
| sampler_name | str | 采样器, 默认 `default_sampler` |
| scheduler_name | str | 调度器, 默认 `default_scheduler` |
| overwrite_step | int | Forced Overwrite of Sampling Step, 默认 -1, 范围 -1-200 |
| overwrite_switch | int | Forced Overwrite of Refiner Switch Step, 默认 -1, 范围 -1-200 |
| overwrite_width | int | Forced Overwrite of Generating Width, 默认 -1, 范围 -1-2048 |
| overwrite_height | int | Forced Overwrite of Generating Height, 默认 -1, 范围 -1-2048 |
| overwrite_vary_strength | float | Forced Overwrite of Denoising Strength of "Vary", 默认 -1, 范围 -1-1.0 |
| overwrite_upscale_strength | float | Forced Overwrite of Denoising Strength of "Upscale", 默认 -1, 范围 -1-1.0 |
| mixing_image_prompt_and_vary_upscale | bool | Mixing Image Prompt and Vary/Upscale, 默认 False |
| mixing_image_prompt_and_inpaint | bool | Mixing Image Prompt and Inpaint, 默认 False |
| debugging_cn_preprocessor | bool | Debug Preprocessors, 默认 False |
| skipping_cn_preprocessor | bool | Skip Preprocessors, 默认 False |
| controlnet_softness | float | Softness of ControlNet, 默认 0.25, 范围 0.0-1.0 |
| canny_low_threshold | int | Canny Low Threshold, 默认 64, 范围 1-255 |
| canny_high_threshold | int | Canny High Threshold, 默认 128, 范围 1-255 |
| freeu_enabled | bool | FreeU enabled, 默认 False |
| freeu_b1 | float | FreeU B1, 默认 1.01 |
| freeu_b2 | float | FreeU B2, 默认 1.02 |
| freeu_s1 | float | FreeU B3, 默认 0.99 |
| freeu_s2 | float | FreeU B4, 默认 0.95 |
| debugging_inpaint_preprocessor | bool | Debug Inpaint Preprocessing, 默认 False |
| inpaint_disable_initial_latent | bool | Disable initial latent in inpaint, 默认 False |
| inpaint_engine | str | Inpaint Engine, 默认 `v1` |
| inpaint_strength | float | Inpaint Denoising Strength, 默认 1.0, 范围 0.0-1.0 |
| inpaint_respective_field | float | Inpaint Respective Field, 默认 1.0, 范围 0.0-1.0 |

## lora

| Name | Type | Description |
| ---- | ---- | ----------- |
| model_name | str | 模型名称 |
| weight | float | 权重, 默认 0.5 |

## 响应参数 | response

成功响应：

**async_process: True**

| Name | Type | Description |
| ---- | ---- | ----------- |
| job_id | int | 任务ID |
| job_type | str | 任务类型 |
| job_stage | str | 任务阶段 |
| job_progress | float | 任务进度 |
| job_status | str | 任务状态 |
| job_step_preview | str | 任务预览 |
| job_result | str | 任务结果 |

**async_process: False**

| Name | Type | Description |
| ---- | ---- | ----------- |
| base64 | str | 图片base64编码, 根据 `require_base64` 参数决定是否为 null |
| url | str | 图片url |
| seed | int | 图片种子 |
| finish_reason | str | 任务结束原因 |

失败响应：
