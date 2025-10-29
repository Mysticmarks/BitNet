# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.11.9-slim-bookworm

FROM ${PYTHON_IMAGE} AS base

LABEL org.opencontainers.image.title="bitnet-runtime"
LABEL org.opencontainers.image.description="Runtime image for BitNet inference binaries"
LABEL org.opencontainers.image.source="https://github.com/microsoft/BitNet"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        clang \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY . .

RUN pip install --upgrade pip \
    && pip install build \
    && python -m build --wheel --outdir /tmp/dist

RUN cmake -S . -B build -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build build --config Release --target llama-cli llama-server llama-quantize

FROM ${PYTHON_IMAGE} AS runtime

ENV BITNET_HOME=/opt/bitnet \
    PATH="${BITNET_HOME}/bin:${PATH}"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${BITNET_HOME}

COPY --from=base /tmp/dist/*.whl /tmp/dist/
RUN pip install /tmp/dist/*.whl && rm -rf /tmp/dist

COPY --from=base /build/build/bin ${BITNET_HOME}/bin
COPY run_inference.py run_inference_server.py ${BITNET_HOME}/
COPY docs/deployment.md ${BITNET_HOME}/docs/deployment.md

ENTRYPOINT ["python", "run_inference.py"]
CMD ["--help"]
