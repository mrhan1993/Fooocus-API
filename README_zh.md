[![Docker Image CI](https://github.com/konieshadow/Fooocus-API/actions/workflows/docker-image.yml/badge.svg?branch=main)](https://github.com/konieshadow/Fooocus-API/actions/workflows/docker-image.yml)

[ [English](/README.md) | 中文 ]

- [简介](#简介)
  - [Fooocus](#fooocus)
  - [Fooocus-API](#fooocus-api)
- [开始](#开始)
  - [在 Replicate 上运行](#在-replicate-上运行)
  - [自托管](#自托管)
    - [conda](#conda)
    - [venv](#venv)
    - [预下载及安装](#预下载及安装)
    - [已经有安装好的 Fooocus](#已经有安装好的-fooocus)
  - [使用Docker启动](#使用docker启动)
- [命令行参数](#命令行参数)
- [更新日志](#更新日志)
- [Apis](#apis)
- [License](#license)
- [感谢 :purple\_heart:](#感谢-purple_heart)


> 注意：
>
> 尽管我进行了测试，但我仍建议你在正式更新前再测一遍
>
> Fooocus 2.5 包含大量更新，其中多数依赖进行了升级，因此，更新后请不要使用 `--skip-pip`. 除非你已经进行过手动更新
>
> 此外, `groundingdino-py` 可能会遇到安装错误, 特别是在中文 windows 环境中, 解决办法参考: [issues](https://github.com/IDEA-Research/GroundingDINO/issues/206)

> 和 DescribeImage 一样，GenerateMask 不会作为 task 处理而是直接返回结果

# ImageEnhance 接口的使用说明

以下面的参数为例，它包含了 ImageEnhance 所需要的主要参数，V1 接口采用和 ImagePrompt 类似的方式将 enhance 控制器拆分成表单形式：

```python
{
  "enhance_input_image": "",
  "enhance_checkbox": true,
  "enhance_uov_method": "Vary (Strong)",
  "enhance_uov_processing_order": "Before First Enhancement",
  "enhance_uov_prompt_type": "Original Prompts",
  "save_final_enhanced_image_only": true,
  "enhance_ctrlnets": [
    {
      "enhance_enabled": false,
      "enhance_mask_dino_prompt": "face",
      "enhance_prompt": "",
      "enhance_negative_prompt": "",
      "enhance_mask_model": "sam",
      "enhance_mask_cloth_category": "full",
      "enhance_mask_sam_model": "vit_b",
      "enhance_mask_text_threshold": 0.25,
      "enhance_mask_box_threshold": 0.3,
      "enhance_mask_sam_max_detections": 0,
      "enhance_inpaint_disable_initial_latent": false,
      "enhance_inpaint_engine": "v2.6",
      "enhance_inpaint_strength": 1,
      "enhance_inpaint_respective_field": 0.618,
      "enhance_inpaint_erode_or_dilate": 0,
      "enhance_mask_invert": false
    }
  ]
}
```

- enhance_input_image：需要增强的图像，如果是 v2 接口，可以提供一个图像 url，必选
- enhance_checkbox：总开关，使用 enhance image 必须设置为 true
- save_final_enhanced_image_only：图像增强是一个管道作业，因此会产生多个结果图像，使用该参数仅返回最终图像

有三个和 UpscaleVary 相关的参数，其作用是执行增强之前或完成增强之后执行 Upscale 或 Vary

- enhance_uov_method：和 UpscaleOrVary 接口一样，Disabled 是关闭
- enhance_uov_processing_order：在增强之前处理还是处理增强后的图像
- enhance_uov_prompt_type：我也不知道具体作用，对着 WebUI 研究研究🧐

`enhance_ctrlnets` 元素为 ImageEnhance 控制器对象列表，该列表最多包含 3 个元素，多余会被丢弃。参数和 WebUI 基本一一对应，需要注意的参数是：

- enhance_enabled：参数控制该 enhance 控制器是否工作，如果没有开启的 enhance 控制器，任务会被跳过
- enhance_mask_dino_prompt：该参数必选，表示需要增强的部位，如果该参数为空，即便 enhance 控制器处于开启状态，也会跳过

# 简介

使用 FastAPI 构建的 [Fooocus](https://github.com/lllyasviel/Fooocus) 的 API。

当前支持的 Fooocus 版本: [2.5.3](https://github.com/lllyasviel/Fooocus/blob/main/update_log.md)。

## Fooocus

**该章节来自 [Fooocus](https://github.com/lllyasviel/Fooocus) 项目。**

Fooocus 是一个图像生成软件 (基于 [Gradio](https://www.gradio.app/))。

Fooocus 是对于 Stable Diffusion 和 Midjourney 的重新思考以及设计：

- 我们学习了 Stable Diffusion 的开源、免费、离线运行。

- 我们学习了 Midjourney 的专注，不需要手动调整，专注于描述词以及图像。

Fooocus 包含了许多内部优化以及质量改进。 忘记那些复杂困难的技术参数，享受人机交互带来的想象力的突破以及探索新的思维

## Fooocus-API

可能您已经尝试过通过 [Gradio 客户端](https://www.gradio.app/docs/client) 来接入 Fooocus，但您可能发现体验并不理想。

Fooocus API 是基于 [FastAPI](https://fastapi.tiangolo.com/) 构建的一系列 `REST` 接口，它们使得利用 Fooocus 的强大功能变得简单易行。现在，您可以使用任何您喜欢的编程语言来轻松地与 Fooocus 进行交互。

此外，我们还提供了详尽的 [API 文档](/docs/api_doc_zh.md) 和丰富的 [示例代码](/examples)，以帮助您快速上手和深入了解如何有效地利用 Fooocus。

# 开始

## 在 Replicate 上运行

现在你可以在 Replicate 上使用 Fooocus-API，在这儿： [konieshadow/fooocus-api](https://replicate.com/konieshadow/fooocus-api).

使用预先调整参数的:

- [konieshadow/fooocus-api-anime](https://replicate.com/konieshadow/fooocus-api-anime)
- [konieshadow/fooocus-api-realistic](https://replicate.com/konieshadow/fooocus-api-realistic)

我认为这是更简单的方法来体验 Fooocus 的强大

> 出于某些原因，上述 replicate 上的实例版本无法更新，你可以参照 [push-a-model](https://replicate.com/docs/guides/push-a-model) 部署自己专用的实例。

## 自托管

需要 Python >= 3.10，或者使用 conda、venv 创建一个新的环境

硬件需求来源于 Fooocus。 详细要求可以看[这里](https://github.com/lllyasviel/Fooocus#minimal-requirement)

### conda

按照下面的步骤启动一个 app：

```shell
conda env create -f environment.yaml
conda activate fooocus-api
```

然后，执行 `python main.py` 启动 app ，默认情况下会监听在 `http://127.0.0.1:8888`

> 如果是第一次运行，程序会自动处理完成剩余的环境配置、模型下载等工作，因此会等待一段时间。也可以预先配置好环境、下载模型，后面会提到。

### venv

和使用 conda 类似，创建虚拟环境，启动 app ，等待程序完成环境安装、模型下载

```powershell
# windows
python -m venv venv
.\venv\Scripts\Activate
```

```shell
# linux
python -m venv venv
source venv/bin/activate
```
然后执行 `python main.py`

### 预下载及安装

如果想要手动配置环境以及放置模型，可以参考下面的步骤

在创建完 conda 或者 venv 环境之后，按照下面的步骤手动配置环境、下载模型

首先，安装 requirements： `pip install -r requirements.txt`

然后安装 pytorch+cuda： `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121`

更多安装信息在 pytorch 官方的 [previous-versions](https://pytorch.org/get-started/previous-versions/) 页面找到。

> 关于 pytorch 和 cuda 的版本，Fooocus API 使用的是 Fooocus 推荐的版本，目前是 pytorch2.1.0+cuda12.1。如果你是个 "犟种" 非要用其他版本，我测试过也是可以的，不过启动的时候记得加上 `--skip-pip`，否则程序会自动替换为推荐版本。

进入 `repositories` 的目录，下载的模型放到这个目录 `repositories\Fooocus\models`。如果你有一个已经安装完成的 Fooocus，在[这里](#已经有安装好的-fooocus)查看如何复用模型

这里是一个启动必须下载的模型列表 (也可能不一样如果 [启动参数](#命令行参数) 不同的话):

- checkpoint: 放到 `repositories\Fooocus\models\checkpoints`
    + [juggernautXL_v8Rundiffusion.safetensors](https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors)

- vae_approx: 放到 `repositories\Fooocus\models\vae_approx`
    + [xlvaeapp.pth](https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth)
    + [vaeapp_sd15.pth](https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt)
    + [xl-to-v1_interposer-v3.1.safetensors](https://huggingface.co/lllyasviel/misc/resolve/main/xl-to-v1_interposer-v3.1.safetensors)

- lora: 放到 `repositories\Fooocus\models\loras`
    + [sd_xl_offset_example-lora_1.0.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors?download=true)

> 国内不好下的到 [这儿](https://www.123pan.com/s/dF5A-SIQsh.html)下载， 提取码: `D4Mk`

### 已经有安装好的 Fooocus

如果你已经有一个安装好的且运行正常的 Fooocus， 推荐的方式是复用模型, 只需要将 Fooocus 根目录下的 `config.txt` 文件复制到 Fooocus API 的根目录即可。 查看 [Customization](https://github.com/lllyasviel/Fooocus#customization) 获取更多细节.

使用这种方法 Fooocus 和 Fooocus API 会同时存在，独立运行互不干扰。

> 不要将已安装的 Fooocus 目录复制到 repositories 目录。

## 使用Docker启动

开始之前，先安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，这是 Docker 可以使用 GPU 的前提。

运行

```shell
docker run -d --gpus=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -p 8888:8888 konieshadow/fooocus-api
```

一个更实用的例子:

```shell
mkdir ~/repositories
mkdir -p ~/.cache/pip

docker run -d --gpus=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v ~/repositories:/app/repositories \
    -v ~/.cache/pip:/root/.cache/pip \
    -p 8888:8888 konieshadow/fooocus-api
```

这里把 `repositories` 和 `pip cache` 映射到了本地

你还可以添加 `-e PIP_INDEX_URL={pypi-mirror-url}` 选项来更换 pip 源

> 0.4.0.0 版本开始，镜像包含完整运行环境，因此只需要根据需要将 `models` 或者项目根目录进行映射即可
> 比如：
> ```
> docker run -d --gpus all \
>     -v /Fooocus-API:/app \
>     -p 8888:8888 konieshadow/fooocus-api
>```

# 命令行参数

- `-h, --help` 显示本帮助并退出
- `--port PORT` 设置监听端口，默认：8888
- `--host HOST` 设置监听地址，默认：127.0.0.1
- `--base-url BASE_URL` 设置返回结果中的地址，默认是： http://host:port
- `--log-level LOG_LEVEL` Uvicorn 中的日志等级，默认：info
- `--skip-pip` 跳过启动时的 pip 安装
- `--preload-pipeline` 启动 http server 之前加载 pipeline
- `--queue-size QUEUE_SIZE` 工作队列大小，默认是 100 ，超过队列的请求会返回失败
- `--queue-history QUEUE_HISTORY` 保留的作业历史，默认 0 即无限制，超过会被删除，包括生成的图像
- `--webhook-url WEBHOOK_URL` 通知生成结果的 webhook 地址，默认为 None
- `--persistent` 持久化历史记录到SQLite数据库，默认关闭
- `--apikey APIKEY` 设置 apikey 以启用安全api，默认值：无（当服务暴露在非本机时强烈建议设置）

> **安全说明：** 远程 URL 输入（`input_image`/`input_mask`/`cn_img*`/`enhance_input_image`/`loras_custom_urls`/`webhook_url`）只会访问公网地址——私网、链路本地（含云元数据）、回环等非公网地址均会被拦截。API key 认证**默认关闭**；当服务可被 `127.0.0.1` 之外的地址访问时，请务必设置 `--apikey`。

从 v0.3.25 开始, Fooocus 的命令行选项也被支持，你可以在启动时加上 Fooocus 支持的选项

比如(需要更大的显存):

```
python main.py --all-in-fp16 --always-gpu
```

完成的 Fooocus 命令行选项可以在[这儿](https://github.com/lllyasviel/Fooocus?tab=readme-ov-file#all-cmd-flags)找到。


# 更新日志

[CHANGELOG](./docs/change_logs_zh.md)

更早的日志可以在 [release page](https://github.com/konieshadow/Fooocus-API/releases) 找到


# Apis

你可以在[这里](/docs/api_doc_zh.md)找到所有的 API 细节

# License

This repository is licensed under the [GUN General Public License v3.0](https://github.com/mrhan1993/Fooocus-API/blob/main/LICENSE)

The default checkpoint is published by [RunDiffusion](https://huggingface.co/RunDiffusion), is licensed under the [CreativeML Open RAIL-M](https://github.com/mrhan1993/Fooocus-API/blob/main/CreativeMLOpenRAIL-M).

or, you can find it [here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)


# 感谢 :purple_heart:

感谢所有为改进 Fooocus API 做出贡献和努力的人。再次感谢 :sparkles: 社区万岁 :sparkles:!
