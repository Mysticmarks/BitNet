# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=debian:12-slim

FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        clang \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN python3 -m venv /opt/bitnet-venv \
    && . /opt/bitnet-venv/bin/activate \
    && pip install --upgrade pip \
    && pip install build \
    && python -m build --wheel --outdir /tmp/dist \
    && cmake -S . -B build -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build build --config Release --target llama-cli llama-server llama-quantize

FROM ${BASE_IMAGE} AS runtime

ENV BITNET_HOME=/opt/bitnet \
    PATH="${BITNET_HOME}/bin:${PATH}"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=base /opt/bitnet-venv /opt/bitnet-venv
RUN python3 -m venv --upgrade /opt/bitnet-venv

WORKDIR ${BITNET_HOME}
COPY --from=base /tmp/dist/*.whl /tmp/dist/
RUN . /opt/bitnet-venv/bin/activate \
    && pip install /tmp/dist/*.whl \
    && rm -rf /tmp/dist

COPY --from=base /workspace/build/bin ${BITNET_HOME}/bin
COPY run_inference.py run_inference_server.py ${BITNET_HOME}/

ENTRYPOINT ["/opt/bitnet-venv/bin/python", "run_inference.py"]
CMD ["--help"]
