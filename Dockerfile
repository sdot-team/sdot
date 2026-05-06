# syntax=docker/dockerfile:1
# Image de développement (Ubuntu). Pour les wheels manylinux, cibuildwheel
# gère ses propres conteneurs — ce Dockerfile sert au dev local et aux tests.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    clang \
    llvm \
    libssl-dev \
    pkg-config \
    unzip \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# xmake via script officiel (le package PyPI "xmake" est sans rapport)
RUN curl -fsSL https://xmake.io/shget.lua | bash -s -- --prefix=/usr/local

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

CMD ["bash"]
