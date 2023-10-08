FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

ARG REGISTER https://pypi.org/simple

RUN pip3 install -i $REGISTER virtualenv
RUN mkdir /app

WORKDIR /app

RUN virtualenv /venv
RUN . /venv/bin/activate && \
    pip3 install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN . /venv/bin/activate && \
    pip3 install -i $REGISTER -r requirements.txt

RUN . /venv/bin/activate && \
    pip3 install -i $REGISTER torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

RUN . /venv/bin/activate && \
    pip3 install -i $REGISTER xformers

COPY . /app/

CMD . /venv/bin/activate && exec python3 main.py --host 0.0.0.0 --port 8888
