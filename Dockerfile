# Use CUDA 12.1 base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda and set up Python environment
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

# Install PyTorch with CUDA 12.1 support
RUN conda install -y python=3.10 \
    && conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia \
    && conda clean -ya

# Install other Python packages
RUN pip install --no-cache-dir \
    Pillow fonttools onnx-coreml coremltools aiohttp aiosignal \
    attrs certifi charset-normalizer discord-py python-dotenv \
    frozenlist idna multidict pynacl requests urllib3 yarl \
    async-timeout \
    https://github.com/ytdl-org/ytdl-nightly/archive/2023.08.07.tar.gz \
    bitsandbytes \
    flash-attn \
    git+https://github.com/huggingface/transformers.git \
    git+https://github.com/huggingface/diffusers.git \
    git+https://github.com/huggingface/peft.git \
    git+https://github.com/huggingface/accelerate.git \
    langchain==0.1.0

# Set up a non-root user
RUN useradd -m -s /bin/bash appuser
# Create .cache directory with correct permissions
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser/.cache
USER appuser

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=appuser:appuser . /app

# Create a startup script
RUN echo '#!/bin/bash\n\
if command -v nvidia-smi &> /dev/null; then\n\
    nvidia-smi\n\
else\n\
    echo "nvidia-smi not found. CUDA may not be available."\n\
fi\n\
python -c "import torch; print('\''CUDA available:'\'', torch.cuda.is_available()); print('\''CUDA version:'\'', torch.version.cuda); print('\''PyTorch version:'\'', torch.__version__)"\n\
exec "$@"' > /app/startup.sh \
&& chmod +x /app/startup.sh

# Set entrypoint to our startup script
ENTRYPOINT ["/app/startup.sh"]

# Set default command
CMD ["python", "Bot.py"]