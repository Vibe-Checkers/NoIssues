FROM docker.io/library/ubuntu:22.04@sha256:104ae83764a5119017b8e8d6218fa0832b09df65aae7d5a6de29a85d813da2fb
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --fix-missing \
        build-essential \
        cmake \
        ninja-build \
        python3 \
        python3-pip \
        python3-dev \
        zlib1g-dev \
        libicu-dev \
        libkrb5-dev \
    && rm -rf /var/lib/apt/lists/*