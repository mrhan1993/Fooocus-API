FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt-get update -y && \
	apt-get install -y curl libgl1 libglib2.0-0 python3-pip python-is-python3 git wget && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app/repositories/Fooocus/models/checkpoints
RUN wget -O illustrious-xl.safetensors https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors
RUN wget -O animaPencilXL_v500.safetensors https://huggingface.co/misri/animaPencilXL_v500/resolve/refs%2Fpr%2F1/animaPencilXL_v500.safetensors

WORKDIR /app/repositories/Fooocus/models/loras
# RUN wget -O otti.safetensors https://huggingface.co/AdiCakepLabs/otti_v1/resolve/main/otti.safetensors
RUN wget -O otti_v2_000002.safetensors https://huggingface.co/AdiCakepLabs/otti_v2/resolve/main/otti_v2-000002.safetensors

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir opencv-python-headless -i https://pypi.org/simple

EXPOSE 3002

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "3002", "--skip-pip"]
