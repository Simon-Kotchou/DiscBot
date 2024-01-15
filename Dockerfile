# Ref https://docs.rapids.ai/install
FROM nvcr.io/nvidia/rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10

# Run everything as root
USER root

# Set our locale to en_US.UTF-8.
ENV LANG en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install python packages per your requirements
RUN pip3 install --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Install software-properties-common
RUN apt update && apt install -y software-properties-common
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
    apt-get update && \
    apt-get install -y libcudnn8 && \
    apt-get install -y libcudnn8-dev && \
    apt-get install -y kmod && \
    apt-get install -y nvidia-cuda-toolkit

# Download and install CuDNN
#RUN wget https://developer.nvidia.com/rdp/cudnn-download -O cudnn-local-repo.deb && \
    #dpkg -i cudnn-local-repo.deb && \ 
RUN apt-get update && apt-get install -y libcudnn8=8.9.5.29-1+cuda11.8 --allow-downgrades && \
    apt-get install -y libcudnn8-dev=8.9.5.29-1+cuda11.8 --allow-downgrades && \
    apt-get install -y libcudnn8-samples=8.9.5.29-1+cuda11.8 --allow-downgrades && \
    export LIBRARY_PATH=/usr/lib/cuda/nvvm/libdevice:$LIBRARY_PATH && \
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda

# Setting LIBRARY_PATH to include the path to the libdevice library is necessary for compiling CUDA code that uses the NVVM compiler
ENV LIBRARY_PATH="/usr/lib/cuda/nvvm/libdevice:$LIBRARY_PATH"
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"

RUN apt-get update && apt-get install -y \
    bzip2 \
    libffi-dev \
    openssl \
    sqlite3 libsqlite3-dev \
    tk-dev \
    tzdata \
    xz-utils \
    zlib1g-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Command to run your bot
CMD ["python", "Bot.py"]
























# # Ref https://docs.rapids.ai/install
# FROM nvcr.io/nvidia/rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10

# # Run everything as root
# USER root

# # Set our locale to en_US.UTF-8.
# ENV LANG en_US.UTF-8
# ENV LC_CTYPE en_US.UTF-8

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # install python packages per your requirements
# RUN pip install --no-cache-dir -r requirements.txt

# RUN apt install -y software-properties-common
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
#     mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
#     add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
#     apt-get update && \
#     apt-get install -y libcudnn8 && \
#     apt-get install -y libcudnn8-dev && \
#     apt-get install -y kmod && \
#     apt-get install -y nvidia-cuda-toolkit

# COPY cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb /var

# # install cudnn
# RUN dpkg -i /var/cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb && \
#     cp /var/cudnn-local-repo-ubuntu2204-8.9.5.29/cudnn-local-535C49CB-keyring.gpg /usr/share/keyrings/ && \
#     apt-get update && \
#     apt-get install -y libcudnn8=8.9.5.29-1+cuda11.8 --allow-downgrades && \
#     apt-get install -y libcudnn8-dev=8.9.5.29-1+cuda11.8 --allow-downgrades && \
#     apt-get install -y libcudnn8-samples=8.9.5.29-1+cuda11.8 --allow-downgrades && \
#     export LIBRARY_PATH=/usr/lib/cuda/nvvm/libdevice:$LIBRARY_PATH && \
#     export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda && \
#     cd /root

# # Setting LIBRARY_PATH to include the path to the libdevice library is necessary for compiling CUDA code that uses the NVVM compiler
# ENV LIBRARY_PATH="/usr/lib/cuda/nvvm/libdevice:$LIBRARY_PATH"
# ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"

# RUN apt-get update && apt-get install -y \
#     bzip2 \
#     libffi-dev \
#     openssl \
#     sqlite3 libsqlite3-dev \
#     tk-dev \
#     tzdata \
#     xz-utils \
#     zlib1g-dev \
#     ffmpeg \ 
#     && rm -rf /var/lib/apt/lists/*

# # Command to run your bot
# CMD ["python", "Bot.py"]

