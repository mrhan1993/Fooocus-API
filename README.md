[![Docker Image CI](https://github.com/konieshadow/Fooocus-API/actions/workflows/docker-image.yml/badge.svg?branch=main)](https://github.com/konieshadow/Fooocus-API/actions/workflows/docker-image.yml)

[ English | [中文](/README_zh.md) ]

- [Introduction](#introduction)
  - [Fooocus](#fooocus)
  - [Fooocus-API](#fooocus-api)
- [Get-Start](#get-start)
  - [Run with Replicate](#run-with-replicate)
  - [Self-hosted](#self-hosted)
    - [conda](#conda)
    - [venv](#venv)
    - [predownload and install](#predownload-and-install)
    - [already exist Fooocus](#already-exist-fooocus)
  - [Start with docker](#start-with-docker)
- [cmd flags](#cmd-flags)
- [Change log](#change-log)
- [Apis](#apis)
- [License](#license)
- [Thanks :purple\_heart:](#thanks-purple_heart)

If this is the first time to use it, it is recommended to use a rewritten new project [FooocusAPI](https://github.com/mrhan1993/FooocusAPI)

A migration guide is provided [here](./docs/migrate.md).

# :warning: Compatibility warning :warning:

When upgrading from version 3.x to version 4.0, please read the following incompatibility notes:

1. If you are using an external Fooocus model (that is, the model is not located in the `repositories` directory), delete the `repositories` directory directly, and then update the `git pull`.
2. If not, move the `repositories` directory to any directory, delete the `repositories` directory, then update the `git pull`, and move the `models` directory back to its original location when it is finished.

# Introduction

FastAPI powered API for [Fooocus](https://github.com/lllyasviel/Fooocus).

Currently loaded Fooocus version: [2.3.0](https://github.com/lllyasviel/Fooocus/blob/main/update_log.md).

## Fooocus

This part from [Fooocus](https://github.com/lllyasviel/Fooocus) project.

Fooocus is an image generating software (based on [Gradio](https://www.gradio.app/)).

Fooocus is a rethinking of Stable Diffusion and Midjourney’s designs:

- Learned from Stable Diffusion, the software is offline, open source, and free.

- Learned from Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images.

Fooocus has included and automated lots of inner optimizations and quality improvements. Users can forget all those difficult technical parameters, and just enjoy the interaction between human and computer to "explore new mediums of thought and expanding the imaginative powers of the human species"

## Fooocus-API

I think you must have tried to use [Gradio client](https://www.gradio.app/docs/client) to call Fooocus, which was a terrible experience for me. 

Fooocus API uses [FastAPI](https://fastapi.tiangolo.com/)  provides the `REST` API for using Fooocus. Now, you can use Fooocus's powerful ability in any language you like. 

In addition, we also provide detailed [documentation](/docs/api_doc_en.md) and [sample code](/examples)

# Get-Start

## Run with Replicate

Now you can use Fooocus-API by Replicate, the model is on [konieshadow/fooocus-api](https://replicate.com/konieshadow/fooocus-api).

With preset:

- [konieshadow/fooocus-api-anime](https://replicate.com/konieshadow/fooocus-api-anime)
- [konieshadow/fooocus-api-realistic](https://replicate.com/konieshadow/fooocus-api-realistic)

I believe this is the easiest way to generate image with Fooocus's power.

## Self-hosted

You need python version >= 3.10, or use conda to create a new env.

The hardware requirements are what Fooocus needs. You can find detail [here](https://github.com/lllyasviel/Fooocus#minimal-requirement)

### conda

You can easily start app follow this step use conda:

```shell
conda env create -f environment.yaml
conda activate fooocus-api
```

and then, run `python main.py` to start app, default, server is listening on `http://127.0.0.1:8888`

> If you are running the project for the first time, you may have to wait for a while, during which time the program will complete the rest of the installation and download the necessary models. You can also do these steps manually, which I'll mention later.

### venv

Similar to using conda, create a virtual environment, and then start and wait for a while

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
and then, run `python main.py`

### predownload and install

If you want to deal with environmental problems manually and download the model in advance, you can refer to the following steps

After creating a complete environment using conda or venv, you can manually complete the installation of the subsequent environment, just follow

first, install requirements: `pip install -r requirements.txt`

then, pytorch with cuda: `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121` , you can find more info about this [here](https://pytorch.org/get-started/previous-versions/),

> It is important to note that for pytorch and cuda versions, the recommended version of Fooocus is used, which is currently pytorch2.1.0+cuda12.1. If you insist, you can also use other versions, but you need to add `--skip-pip` when you start app, otherwise the recommended version will be installed automatically

Go to the `repositories` directories, download models and put it into `repositories\Fooocus\models`

If you have Fooocus installed, see [already-exist-fooocus](#already-exist-fooocus)

here is a list need to download for startup (for different [startup params](#cmd-flags) maybe difference):

- checkpoint:  path to `repositories\Fooocus\models\checkpoints`
    + [juggernautXL_version6Rundiffusion.safetensors](https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_version6Rundiffusion.safetensors)

- vae_approx: path to `repositories\Fooocus\models\vae_approx`
    + [xlvaeapp.pth](https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth)
    + [vaeapp_sd15.pth](https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt)
    + [xl-to-v1_interposer-v3.1.safetensors](https://huggingface.co/lllyasviel/misc/resolve/main/xl-to-v1_interposer-v3.1.safetensors)

- lora: path to `repositories\Fooocus\models\loras`
    + [sd_xl_offset_example-lora_1.0.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors?download=true)

> I've uploaded the model I'm using, which contains almost all the base models that Fooocus will use! I put it [here](https://www.123pan.com/s/dF5A-SIQsh.html) 提取码: `D4Mk`

### already exist Fooocus

If you already have Fooocus installed, and it is work well, The recommended way is to reuse models, you just simple copy `config.txt` file from your local Fooocus folder to Fooocus-API root folder. See [Customization](https://github.com/lllyasviel/Fooocus#customization) for details.

Use this method you will have both Fooocus and Fooocus-API running at the same time. And they operate independently and do not interfere with each other.

> Do not copy Fooocus to repositories directory

## Start with docker

Before use docker with GPU, you should [install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first.

Run

```shell
docker run -d --gpus=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -p 8888:8888 konieshadow/fooocus-api
```

For a more complex usage:

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

It will be persistent the dependent repositories and pip cache.

You can add `-e PIP_INDEX_URL={pypi-mirror-url}` to docker run command to change pip index url.

> From version 0.4.0.0, Full environment include in docker image, mapping `models` or project root if you needed
> For example:
> ```
> docker run -d --gpus all \
>     -v /Fooocus-API:/app \
>     -p 8888:8888 konieshadow/fooocus-api
>```

# cmd flags

- `-h, --help` show this help message and exit
- `--port PORT` Set the listen port, default: 8888
- `--host HOST` Set the listen host, default: 127.0.0.1
- `--base-url BASE_URL` Set base url for outside visit, default is http://host:port
- `--log-level LOG_LEVEL` Log info for Uvicorn, default: info
- `--skip-pip` Skip automatic pip install when setup
- `--preload-pipeline` Preload pipeline before start http server
- `--queue-size QUEUE_SIZE` Working queue size, default: 100, generation requests exceeding working queue size will return failure
- `--queue-history QUEUE_HISTORY` Finished jobs reserve size, tasks exceeding the limit will be deleted, including output image files, default: 0, means no limit
- `--webhook-url WEBHOOK_URL` Webhook url for notify generation result, default: None
- `--persistent` Store history to db
- `--apikey APIKEY` Set apikey to enable secure api, default: None

Since v0.3.25, added CMD flags support of Fooocus. You can pass any argument which Fooocus supported.

For example, to startup image generation (need more vRAM):

```
python main.py --all-in-fp16 --always-gpu
```

For Fooocus CMD flags, see [here](https://github.com/lllyasviel/Fooocus?tab=readme-ov-file#all-cmd-flags).


# Change log

**[24/01/31] v0.3.30** : Add secure API future, use `--apikey APIKEY` to set apikey at startup

**[24/01/26] v0.3.30** : Optimize the underlying logic of task execution, default queue size to 100

**[24/01/10] v0.3.29** : support for store history to db

**[24/01/09] v0.3.29** : Image Prompt Mixing requirements implemented, With this implementation, you can send image prompts, and perform inpainting or upscale with a single request.

**[24/01/04] v0.3.29** : Merged Fooocus v2.1.860

**[24/01/03] v0.3.28** : add text-to-image-with-ip interface

**[23/12/29] v0.3.27** : Add describe interface，now you can get prompt from image

**[23/12/29] v0.3.27** : Add query job history api. Add webhook_url support for each generation request.

**[23/12/28] v0.3.26** : **Break Change**: Add web-hook cmd flag for notify generation result. Change async job id to uuid to avoid conflict between each startup.

**[23/12/22] v0.3.25** : Add CMD flags support of Fooocus. **Break Change**: Removed cli argument `disable-private-log`. You can use Fooocus's `--disable-image-log` for the same purpose.

older change history you can find in [release page](https://github.com/konieshadow/Fooocus-API/releases)


# Apis

you can find all api detail [here](/docs/api_doc_en.md)

# License

This repository is licensed under the [GUN General Public License v3.0](https://github.com/mrhan1993/Fooocus-API/blob/main/LICENSE)

The default checkpoint is published by [RunDiffusion](https://huggingface.co/RunDiffusion), is licensed under the [CreativeML Open RAIL-M](https://github.com/mrhan1993/Fooocus-API/blob/main/CreativeMLOpenRAIL-M).

or, you can find it [here](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

# Thanks :purple_heart:

Thanks for all your contributions and efforts towards improving the Fooocus API. We thank you for being part of our :sparkles: community :sparkles:!
