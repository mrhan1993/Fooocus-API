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

如果是第一次使用，推荐使用重写后的新项目 [FooocusAPI](https://github.com/mrhan1993/FooocusAPI)

我还准备了一个[迁移指南](./docs/migrate_zh.md)

# :warning: 兼容性警告 :warning:

如果是从 0.3.x 版本升级到 0.4.0 版本，请务必阅读以下兼容性说明：

1. 如果你使用的是外部 Fooocus 模型（即模型不是位于 `repositories/Fooocus/models` 目录下），直接删除 `repositories` 目录，然后执行 `git pull` 更新即可
2. 如果不是上述方式，将 `repositories/Fooocus/models` 目录移动到任意目录，删除 `repositories` 目录，然后执行 `git pull` 更新，完成后将 `models` 目录移动回原位置

# 简介

使用 FastAPI 构建的 [Fooocus](https://github.com/lllyasviel/Fooocus) 的 API。

当前支持的 Fooocus 版本: [2.3.1](https://github.com/lllyasviel/Fooocus/blob/main/update_log.md)。

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
- `--apikey APIKEY` 设置 apikey 以启用安全api，默认值：无

从 v0.3.25 开始, Fooocus 的命令行选项也被支持，你可以在启动时加上 Fooocus 支持的选项

比如(需要更大的显存):

```
python main.py --all-in-fp16 --always-gpu
```

完成的 Fooocus 命令行选项可以在[这儿](https://github.com/lllyasviel/Fooocus?tab=readme-ov-file#all-cmd-flags)找到。


# 更新日志

**[24/01/31] v0.3.30** : 增加接口认证功能, 可以通过启动参数 `--apikey APIKEY` 来设置 apikey

**[24/01/26] v0.3.30** : 优化任务执行底层逻辑。调整默认队列大小为100

**[24/01/10] v0.3.29** : 支持将历史生成数据持久化到数据库，并且支持从数据库中读取历史数据

**[24/01/09] v0.3.29** : Image Prompt Mixing requirements implemented, With this implementation, you can send image prompts, and perform inpainting or upscale with a single request.

**[24/01/04] v0.3.29** : 合并了 Fooocus v2.1.860

**[24/01/03] v0.3.28** : 增加 text-to-image-with-ip 接口

**[23/12/29] v0.3.27** : 增加 describe 接口，现在你可以使用图像反推提示词了

**[23/12/29] v0.3.27** : 增加查询历史 API。增加 webhook_url 对所有请求的支持

**[23/12/28] v0.3.26** : **重大变更**: 添加 webhook 选项以支持生成完毕后的事件通知。将 async 的任务 ID 由数字改为 UUID 来避免应用重启后造成的混乱

**[23/12/22] v0.3.25** : 增加对 Fooocus 命令行选项的支持 **重大变更**: 移除 `disable-private-log` 选项，你可以使用 Fooocus 原生的 `--disable-image-log` 来达到同样的效果

更早的日志可以在 [release page](https://github.com/konieshadow/Fooocus-API/releases) 找到


# Apis

你可以在[这里](/docs/api_doc_zh.md)找到所有的 API 细节

# License

This repository is licensed under the [GUN General Public License v3.0](https://github.com/mrhan1993/Fooocus-API/blob/main/LICENSE)

The default checkpoint is published by [RunDiffusion](https://huggingface.co/RunDiffusion), is licensed under the [CreativeML Open RAIL-M](https://github.com/mrhan1993/Fooocus-API/blob/main/CreativeMLOpenRAIL-M).

or, you can find it [here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)


# 感谢 :purple_heart:

感谢所有为改进 Fooocus API 做出贡献和努力的人。再次感谢 :sparkles: 社区万岁 :sparkles:!
