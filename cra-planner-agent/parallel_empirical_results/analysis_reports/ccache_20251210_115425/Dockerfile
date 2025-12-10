FROM ubuntu:22.04

# Install essential build and runtime packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        wget \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-setuptools \
        pkg-config \
        libffi-dev \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libgdbm-dev \
        liblzma-dev \
        libxml2-dev \
        libxslt1-dev \
        libjpeg-dev \
        libpng-dev \
        gcc \
        g++ \
        make \
        openjdk-11-jdk \
        maven \
        nodejs \
        npm \
        -y && \
    rm -rf /var/lib/apt/lists/*

# Prepare tmp workspace for optional dependencies
RUN mkdir -p /tmp

# Optional Python dependencies if provided in build context
RUN if [ -s /tmp/requirements.txt ]; then \
        python3 -m pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi

# Optional Node dependencies if provided in build context
RUN if [ -s /tmp/package.json ]; then \
        if [ -f /tmp/package-lock.json ]; then \
            npm ci --silent --no-audit --no-fund; \
        else \
            npm install --silent --no-audit --no-fund; \
        fi; \
    fi

# Expose port (adjust as needed)
EXPOSE 8080

# Default command
CMD ["bash"]