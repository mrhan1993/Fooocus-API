# Fooocus-API

[![Docker Image CI](https://github.com/konieshadow/Fooocus-API/actions/workflows/docker-image.yml/badge.svg?branch=main)](https://github.com/konieshadow/Fooocus-API/actions/workflows/docker-image.yml)

FastAPI powered API for [Fooocus](https://github.com/lllyasviel/Fooocus)

Currently loaded Fooocus version: 2.1.44

### Run with Replicate
Now you can use Fooocus-API by Replicate, the model is in [konieshadow/fooocus-api](https://replicate.com/konieshadow/fooocus-api).

I believe this is the easiest way to generate image with Fooocus's power.

### Install dependencies.
Need python version >= 3.10
```
pip install -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118 xformers
```
You may change the part "cu118" of extra-index-url to your local installed cuda driver version.

### Sync dependent and download models (Optional)
```
python main.py --sync-repo only
```
After run successful, you can see the terminal print where to put the model files for Fooocus.

Then you can put the model files to target directories manually, or let it auto downloads when start app.

It will also apply user_path_config.txt config file as Fooocus. See [Changing Model Path](https://github.com/lllyasviel/Fooocus#changing-model-path).

### Start app
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
