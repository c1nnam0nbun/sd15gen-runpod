# SD 1.5 + CivitAI LoRA — RunPod Serverless

Run Stable Diffusion 1.5 with any CivitAI LoRA on a RunPod serverless endpoint.  
The LoRA is downloaded fresh each invocation (cached in `/tmp` for the worker's lifetime).

---

## Project structure

```
.
├── handler.py        # RunPod worker — loads SD 1.5, downloads LoRA, generates image
├── Dockerfile        # Bakes SD 1.5 into the image for fast cold starts
├── requirements.txt  # Python dependencies
├── client.py         # CLI client to submit jobs and save the result PNG
├── .env.example      # Environment variable template
└── README.md
```

---

## 1 — Prerequisites

- Docker (to build & push the image)
- A [RunPod](https://runpod.io) account
- A [CivitAI](https://civitai.com) account + API key
- A Docker Hub (or any registry) account

---

## 2 — Build & push the Docker image

```bash
# Build (bakes SD 1.5 weights into the image — ~6 GB)
docker build -t your-dockerhub-user/sd15-lora-runpod:latest .

# Push
docker push your-dockerhub-user/sd15-lora-runpod:latest
```

> **Tip:** If you want a smaller image and don't mind a slower cold start,
> remove the `RUN python …` block from the Dockerfile that pre-downloads the model.

---

## 3 — Create a RunPod Serverless Endpoint

1. Go to **RunPod → Serverless → + New Endpoint**
2. Set **Container Image** to `your-dockerhub-user/sd15-lora-runpod:latest`
3. Choose a GPU — **RTX 3090 / 4090 / A4000** all work well
4. Under **Environment Variables**, add:
   - `CIVITAI_API_KEY` → your CivitAI API key  
   *(do NOT add RUNPOD_API_KEY here — that's client-side only)*
5. Set **Container Disk** to at least `10 GB`
6. Save → copy your **Endpoint ID**

---

## 4 — Run a generation

```bash
# Install client deps
pip install requests

# Set your RunPod API key
export RUNPOD_API_KEY=your_runpod_api_key

# Basic generation (no LoRA)
python client.py \
  --endpoint YOUR_ENDPOINT_ID \
  --prompt "a photo of an astronaut riding a horse on mars, cinematic"

# With a CivitAI LoRA  (use the version ID from the URL: ?modelVersionId=XXXXX)
python client.py \
  --endpoint YOUR_ENDPOINT_ID \
  --prompt "a portrait of a woman in the style of sks, masterpiece" \
  --lora 12345 \
  --lora-scale 0.8 \
  --steps 30 \
  --width 512 \
  --height 768 \
  --output portrait.png
```

---

## 5 — Input schema

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | Positive prompt |
| `negative_prompt` | string | `"low quality, blurry, bad anatomy"` | Negative prompt |
| `lora_version_id` | string | `null` | CivitAI model **version** ID |
| `lora_scale` | float 0–2 | `0.8` | LoRA strength |
| `steps` | int 1–150 | `30` | Inference steps |
| `guidance_scale` | float | `7.5` | CFG scale |
| `width` | int | `512` | Output width (clamped to 256–768, multiple of 8) |
| `height` | int | `512` | Output height (clamped to 256–768, multiple of 8) |
| `seed` | int | `null` | Set for reproducible results |

### Finding the CivitAI version ID

Open the LoRA page → select the version you want → look at the URL:  
`https://civitai.com/models/12345?modelVersionId=**67890**`  
Use `67890` as your `lora_version_id`.

---

## 6 — Output schema

```json
{
  "image_base64": "<PNG encoded as base64>",
  "nsfw_detected": false,
  "meta": {
    "prompt": "...",
    "negative_prompt": "...",
    "lora_version_id": "67890",
    "lora_scale": 0.8,
    "steps": 30,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 768,
    "seed": null,
    "generation_time_seconds": 4.2
  }
}
```

---

## Notes

- **LoRA caching:** Downloaded LoRAs are stored in `/tmp/loras` for the worker's lifetime.  
  If the same worker handles multiple jobs with the same LoRA, subsequent downloads are skipped.
- **Scheduler:** Uses DPM-Solver++ with Karras sigmas — good quality at 20–30 steps.
- **Safety checker:** Disabled. Add it back in `handler.py` if needed.
