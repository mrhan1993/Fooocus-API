# Fooocus-API

[![Docker Image CI](https://github.com/konieshadow/Fooocus-API/actions/workflows/docker-image.yml/badge.svg?branch=main)](https://github.com/konieshadow/Fooocus-API/actions/workflows/docker-image.yml)

FastAPI powered API for [Fooocus](https://github.com/lllyasviel/Fooocus)

Currently loaded Fooocus version: 2.1.679

### Run with Replicate
Now you can use Fooocus-API by Replicate, the model is in [konieshadow/fooocus-api](https://replicate.com/konieshadow/fooocus-api).

I believe this is the easiest way to generate image with Fooocus's power.

### Colab
The colab notebook uses the Fooocus's `colab` branch, which may lack some latest features.

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konieshadow/Fooocus-API/blob/colab/colab.ipynb) | Fooocus-API

### Start app
Need python version >= 3.10, or use conda to create a new env.

```
conda env create -f environment.yaml
conda activate fooocus-api
```

Set enviroment variable `TORCH_INDEX_URL` to the version corresponding to the local cuda driver.
Default is "https://download.pytorch.org/whl/cu121", you may change the part "cu118".

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

#### Text to Image
> POST /v1/generation/text-to-image

Alternative api for the normal image generation of Fooocus Gradio interface.

Add support for response in PNG bytes directly when request's 'Accept' header is 'image/png'.

#### Image Upscale or Variation
> POST /v1/generation/image-upscale-vary

Alternative api for 'Upscale or Variation' tab of Fooocus Gradio interface.

#### Image Inpaint or Outpaint
> POST /v1/generation/image-inpait-outpaint

Alternative api for 'Inpaint or Outpaint' tab of Fooocus Gradio interface.

#### Image Prompt
> POST /v1/generation/image-prompt

Alternative api for 'Image Prompt' tab of Fooocus Gradio interface.
