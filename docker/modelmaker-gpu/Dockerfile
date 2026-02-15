FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.9 \
        python3.9-distutils \
        python3.9-venv \
    && curl -fsSL https://bootstrap.pypa.io/pip/3.9/get-pip.py -o /tmp/get-pip.py \
    && python3.9 /tmp/get-pip.py \
    && rm -f /tmp/get-pip.py \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements/modelmaker.txt /tmp/requirements/modelmaker.txt
RUN python3.9 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3.9 -m pip install --no-cache-dir -r /tmp/requirements/modelmaker.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace/src

CMD ["/bin/bash"]
