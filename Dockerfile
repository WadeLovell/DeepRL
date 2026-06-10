FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    git ffmpeg libgl1 libglew-dev libosmesa6-dev patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/deep_rl
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install "gymnasium[mujoco]"
COPY . .
RUN pip install -e .
