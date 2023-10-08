FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

ARG REGISTER https://pypi.org/simple

RUN --mount=type=cache,target=/root/.cache/pip pip3 install virtualenv
RUN mkdir /app

WORKDIR /app

RUN virtualenv /venv
RUN . /venv/bin/activate && \
    pip3 install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN . /venv/bin/activate && \
    pip3 install -i $REGISTER -r requirements.txt

RUN . /venv/bin/activate && \
    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118 xformers

COPY . /app/

ENV LISTEN_HOST 127.0.0.1
ENV LISTEN_PORT 8888

ENTRYPOINT [ "bash", "-c", ". /venv/bin/activate && exec \"$@\"", "--" ]
CMD [ "python3", "main.py", "--host", "$LISTEN_HOST", "--port", "$LISTEN_PORT" ]
