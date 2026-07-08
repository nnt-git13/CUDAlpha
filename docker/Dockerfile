# CUDA devel base so nvcc/NVRTC headers are available for CuPy RawKernel builds.
# Match the CUDA major version to your torch/cupy/cuOpt wheels.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-side deps first for layer caching.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# GPU libraries are environment-specific — install the ones matching this base.
# Uncomment / adjust for your CUDA + GPU:
# RUN pip3 install --no-cache-dir cupy-cuda12x nvidia-cuopt

COPY . .
ENV PYTHONPATH=/app

# Default: run the full benchmark sweep. Override for the dashboard service.
CMD ["python3", "-m", "bench.run_all", "--workload", "all"]
