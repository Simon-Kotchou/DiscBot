# Ref https://hub.docker.com/layers/nvidia/cuda/11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG PYTHON_VERSION=3.10

# Run everything as root
USER root

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y --no-install-recommends build-essential \
    vim wget curl bzip2 git unzip g++ binutils cmake locales \
    ca-certificates apt-transport-https gnupg software-properties-common \
    libjpeg-dev libpng-dev iputils-ping net-tools libgl1 libglib2.0-0 \
    libffi-dev openssl sqlite3 libsqlite3-dev tk-dev tzdata xz-utils \
    zlib1g-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN ln -fs /usr/share/zoneinfo/America/Phoenix /etc/localtime && dpkg-reconfigure -f noninteractive tzdata && locale-gen en_US.UTF-8

# Set our locale to en_US.UTF-8.
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION conda-build anaconda-client numpy pyyaml scipy ipython mkl mkl-include \
        cffi ninja setuptools typing_extensions future six requests dataclasses cython typing conda-package-handling && \
    /opt/conda/bin/conda install -c pytorch magma-cuda118 &&  \
    /opt/conda/bin/pip3 install --no-cache-dir Pillow fonttools onnx-coreml coremltools aiohttp aiosignal \
    attrs certifi charset-normalizer discord-py python-dotenv frozenlist idna multidict pynacl requests urllib3 yarl \ 
    async-timeout https://github.com/ytdl-org/ytdl-nightly/archive/2023.08.07.tar.gz \ 
    bitsandbytes git+https://github.com/huggingface/transformers.git \
    git+https://github.com/kashif/diffusers.git@wuerstchen-v3 \ 
    git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/accelerate.git langchain==0.1.0 && \
    /opt/conda/bin/conda clean -ya && \
    /opt/conda/bin/conda clean -y --force-pkgs-dirs
# git+https://github.com/huggingface/diffusers.git
ENV PATH /opt/conda/bin:$PATH
RUN conda config --add channels pytorch

# Set the working directory in the container
WORKDIR /app
COPY . /app

# Command to run your bot
CMD ["python", "Bot.py"]