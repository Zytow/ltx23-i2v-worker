FROM runpod/pytorch:2.6.0-py3.11-cuda12.6.3-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .

ENV MODEL_PATH=/runpod-volume/models/ltx-video-2.3
ENV HF_HOME=/runpod-volume/hf-cache
ENV HUGGING_FACE_HUB_TOKEN=""

CMD ["python", "-u", "handler.py"]
