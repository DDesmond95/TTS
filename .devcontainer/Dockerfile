# ================================
# CUDA base image
# ================================
ARG CUDA_IMAGE_TAG
FROM nvidia/cuda:${CUDA_IMAGE_TAG}

# ================================
# Build arguments
# ================================
ARG USERNAME
ARG USER_UID
ARG USER_GID=${USER_UID}

# ================================
# Environment (non-interactive)
# ================================
ENV DEBIAN_FRONTEND=noninteractive

# ================================
# System packages + Python 3.10 and 3.12
# ================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git openssh-client ca-certificates curl wget bash \
    sox libsox-dev libsox-fmt-all \
    build-essential ninja-build cmake passwd software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv \
    python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Set interpreter defaults:
#   - python  -> Python 3.10
#   - python3 -> Python 3.12
# ================================
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 110 \
    && update-alternatives --set     python  /usr/bin/python3.10 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 120 \
    && update-alternatives --set     python3 /usr/bin/python3.12

# ================================
# Install pip for each interpreter (once per version)
#   - Installs /usr/local/bin/pip3.10 and /usr/local/bin/pip3.12 (typical)
# ================================
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | /usr/bin/python3.10 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | /usr/bin/python3.12

# ================================
# Set pip defaults:
#   - pip  -> Python 3.10 pip
#   - pip3 -> Python 3.12 pip
# ================================
RUN update-alternatives --install /usr/bin/pip  pip  /usr/local/bin/pip3.10 110 \
    && update-alternatives --set     pip  /usr/local/bin/pip3.10 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.12 120 \
    && update-alternatives --set     pip3 /usr/local/bin/pip3.12


# ================================
# Sanity check: verify the mapping is correct
# ================================
RUN python  --version && pip  --version \
    && python3 --version && pip3 --version

# ================================
# Create non-root user
# ================================
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd  --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# ================================
# Passwordless sudo
# ================================
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# ================================
# Environment
#   - Keep user-installed python tools discoverable
# ================================
ENV PATH=/home/${USERNAME}/.local/bin:$PATH
WORKDIR /workspace

# ================================
# Switch user
# ================================
USER ${USERNAME}

# ================================
# Shell
# ================================
SHELL ["/bin/bash", "-c"]
