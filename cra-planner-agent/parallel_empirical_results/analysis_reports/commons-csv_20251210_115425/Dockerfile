FROM ubuntu:22.04@sha256:104ae83764a5119017b8e8d6218fa0832b09df65aae7d5a6de29a85d813da2fb
ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        python3 \
        python3-pip \
        git \
        ca-certificates \
        pkg-config \
        libtool \
        automake \
        autoconf \
        bzip2 \
        zlib1g-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*