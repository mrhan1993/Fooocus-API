# Fooocus-API

[![Docker Image CI](https://github.com/konieshadow/Fooocus-API/actions/workflows/docker-image.yml/badge.svg?branch=main)](https://github.com/konieshadow/Fooocus-API/actions/workflows/docker-image.yml)

FastAPI powered API for [Fooocus](https://github.com/lllyasviel/Fooocus)

Currently loaded Fooocus version: 2.1.728

### Run with Replicate
Now you can use Fooocus-API by Replicate, the model is in [konieshadow/fooocus-api](https://replicate.com/konieshadow/fooocus-api).

I believe this is the easiest way to generate image with Fooocus's power.

### Colab
The colab notebook uses the Fooocus's `colab` branch, which may lack some latest features.

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konieshadow/Fooocus-API/blob/colab/colab.ipynb) | Fooocus-API

### Reuse model files from Fooocus
You can simple copy `user_path_config.txt` file from your local Fooocus folder to Fooocus-API's root folder. See [Customization](https://github.com/lllyasviel/Fooocus#customization) for details.

### Start app
Need python version >= 3.10, or use conda to create a new env.

```
conda env create -f environment.yaml
conda activate fooocus-api
```

Run
```
python main.py
```
On default, server is listening on 'http://127.0.0.1:8888'

For pragram arguments, see
```
python main.py -h
```

### Start with docker
Run
```
docker run --gpus=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -p 8888:8888 konieshadow/fooocus-api
```

For a more complex usage:
```
mkdir ~/repositories
mkdir -p ~/.cache/pip

docker run --gpus=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
    -v ~/repositories:/app/repositories \
    -v ~/.cache/pip:/root/.cache/pip \
    -p 8888:8888 konieshadow/fooocus-api
```
It will persistent the dependent repositories and pip cache.


### Test API
You can open the Swagger Document in "http://127.0.0.1:8888/docs", then click "Try it out" to send a request.

### Completed Apis
Swagger openapi defination see [openapi.json](docs/openapi.json).

You can import it in [Swagger-UI](https://swagger.io/tools/swagger-ui/) editor.

All the generation api support for response in PNG bytes directly when request's 'Accept' header is 'image/png'.

All the generation api support async process by pass parameter `async_process`` to true. And then use query job api to retrieve progress and generation results.

#### Text to Image
> POST /v1/generation/text-to-image

Alternative api for the normal image generation of Fooocus Gradio interface.

#### Image Upscale or Variation
> POST /v1/generation/image-upscale-vary

Alternative api for 'Upscale or Variation' tab of Fooocus Gradio interface.

#### Image Inpaint or Outpaint
> POST /v1/generation/image-inpait-outpaint

Alternative api for 'Inpaint or Outpaint' tab of Fooocus Gradio interface.

#### Image Prompt
> POST /v1/generation/image-prompt

Alternative api for 'Image Prompt' tab of Fooocus Gradio interface.

#### Query Job
> GET /v1/generation/query-job

Query async generation request results, return job progress and generation results.

#### Query Job Queue Info
> GET /v1/generation/job-queue

Query job queue info, include running job count, finished job count and last job id.

#### Get All Model Names
> GET /v1/engines/all-models

Get all filenames of base model and lora.

#### Refresh Models
> POST /v1/engines/refresh-models

Refresh local files and get all filenames of base model and lora.