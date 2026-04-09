import runpod
import torch
import requests
import os
import io
import base64
import logging
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", "")
BASE_MODEL_ID   = os.environ.get("BASE_MODEL_ID", "runwayml/stable-diffusion-v1-5")
LORA_CACHE_DIR  = "/tmp/loras"
os.makedirs(LORA_CACHE_DIR, exist_ok=True)

# ── Load base pipeline once at cold start ─────────────────────────────────────
log.info(f"Loading base model: {BASE_MODEL_ID}")
t0 = time.time()

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

log.info(f"Base model loaded in {time.time() - t0:.1f}s")


# ── Helpers ───────────────────────────────────────────────────────────────────
def download_lora(version_id: str) -> str:
    """
    Download a LoRA from CivitAI by model version ID.
    Caches to /tmp/loras for the lifetime of the worker.
    """
    local_path = os.path.join(LORA_CACHE_DIR, f"{version_id}.safetensors")

    if os.path.exists(local_path):
        log.info(f"LoRA {version_id} already in worker cache, skipping download")
        return local_path

    url = f"https://civitai.com/api/download/models/{version_id}"
    headers = {}
    if CIVITAI_API_KEY:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"

    log.info(f"Downloading LoRA version {version_id} from CivitAI...")
    t0 = time.time()

    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = os.path.getsize(local_path) / 1024 / 1024
    log.info(f"Downloaded {size_mb:.1f} MB in {time.time() - t0:.1f}s → {local_path}")
    return local_path


def image_to_base64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def validate_input(inputs: dict) -> str | None:
    """Return an error string if inputs are invalid, else None."""
    if not inputs.get("prompt"):
        return "Missing required field: prompt"
    steps = inputs.get("steps", 30)
    if not (1 <= int(steps) <= 150):
        return "steps must be between 1 and 150"
    scale = inputs.get("lora_scale", 0.8)
    if not (0.0 <= float(scale) <= 2.0):
        return "lora_scale must be between 0.0 and 2.0"
    return None


# ── Handler ───────────────────────────────────────────────────────────────────
def handler(job):
    inputs = job.get("input", {})
    log.info(f"Job {job['id']} received — inputs: {inputs}")

    # Validate
    err = validate_input(inputs)
    if err:
        return {"error": err}

    # Parse inputs
    prompt          = inputs["prompt"]
    negative_prompt = inputs.get("negative_prompt", "low quality, blurry, bad anatomy")
    lora_version_id = inputs.get("lora_version_id")          # CivitAI version ID (string)
    lora_scale      = float(inputs.get("lora_scale", 0.8))
    steps           = int(inputs.get("steps", 30))
    guidance_scale  = float(inputs.get("guidance_scale", 7.5))
    width           = int(inputs.get("width", 512))
    height          = int(inputs.get("height", 512))
    seed            = inputs.get("seed")                      # optional int

    # Clamp dimensions to multiples of 8 that SD 1.5 handles well
    width  = min(max((width  // 8) * 8, 256), 768)
    height = min(max((height // 8) * 8, 256), 768)

    # Seed
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(int(seed))

    lora_loaded = False

    try:
        # Load LoRA if requested
        if lora_version_id:
            lora_path = download_lora(str(lora_version_id))
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_scale)
            lora_loaded = True
            log.info(f"LoRA {lora_version_id} fused at scale {lora_scale}")

        # Generate
        log.info(f"Generating {width}x{height} image, steps={steps}")
        t0 = time.time()

        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )

        elapsed = time.time() - t0
        log.info(f"Generation done in {elapsed:.1f}s")

        image = output.images[0]
        nsfw  = output.nsfw_content_detected[0] if output.nsfw_content_detected else False

        return {
            "image_base64": image_to_base64(image),
            "nsfw_detected": nsfw,
            "meta": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "lora_version_id": lora_version_id,
                "lora_scale": lora_scale if lora_loaded else None,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "generation_time_seconds": round(elapsed, 2),
            },
        }

    except requests.HTTPError as e:
        log.error(f"CivitAI download failed: {e}")
        return {"error": f"Failed to download LoRA from CivitAI: {e}"}

    except Exception as e:
        log.exception("Unexpected error during generation")
        return {"error": str(e)}

    finally:
        if lora_loaded:
            try:
                pipe.unfuse_lora()
                pipe.unload_lora_weights()
                log.info("LoRA unloaded")
            except Exception:
                pass


runpod.serverless.start({"handler": handler})
