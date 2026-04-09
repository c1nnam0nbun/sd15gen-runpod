FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the base SD 1.5 model into the image so cold starts are fast.
# Remove this block if you'd rather pull from HF at runtime (saves image size).
ARG HF_TOKEN=""
RUN python - <<'EOF'
from diffusers import StableDiffusionPipeline
import torch, os
token = os.environ.get("HF_TOKEN") or None
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    use_auth_token=token,
)
# save to default HF cache so handler.from_pretrained finds it instantly
print("Model cached successfully.")
EOF

# Copy handler last (cache-friendly layering)
COPY handler.py .

CMD ["python", "-u", "handler.py"]
