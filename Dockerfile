FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3 python3-pip python3-virtualenv && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN virtualenv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

COPY . /app/

CMD python3 main.py --host 0.0.0.0 --port 8888
