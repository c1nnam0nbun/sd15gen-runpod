#!/usr/bin/env python3
"""
client.py — Send a generation request to your RunPod serverless endpoint
and save the result as a PNG.

Usage:
    python client.py \
        --endpoint YOUR_ENDPOINT_ID \
        --prompt "a portrait of a woman, masterpiece" \
        --lora 12345 \
        --output out.png
"""

import argparse
import base64
import json
import os
import sys
import time

import requests

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
BASE_URL       = "https://api.runpod.io/v2"


def parse_args():
    p = argparse.ArgumentParser(description="RunPod SD 1.5 + LoRA client")
    p.add_argument("--endpoint",        required=True,  help="RunPod endpoint ID")
    p.add_argument("--prompt",          required=True,  help="Positive prompt")
    p.add_argument("--negative",        default="low quality, blurry, bad anatomy",
                                                        help="Negative prompt")
    p.add_argument("--lora",            default=None,   help="CivitAI model version ID")
    p.add_argument("--lora-scale",      type=float, default=0.8)
    p.add_argument("--steps",           type=int,   default=30)
    p.add_argument("--guidance-scale",  type=float, default=7.5)
    p.add_argument("--width",           type=int,   default=512)
    p.add_argument("--height",          type=int,   default=512)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--output",          default="output.png", help="Where to save the PNG")
    p.add_argument("--timeout",         type=int,   default=300, help="Max wait seconds")
    return p.parse_args()


def run(args):
    if not RUNPOD_API_KEY:
        sys.exit("Set RUNPOD_API_KEY env var first.")

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": {
            "prompt":          args.prompt,
            "negative_prompt": args.negative,
            "steps":           args.steps,
            "guidance_scale":  args.guidance_scale,
            "width":           args.width,
            "height":          args.height,
        }
    }
    if args.lora:
        payload["input"]["lora_version_id"] = args.lora
        payload["input"]["lora_scale"]      = args.lora_scale
    if args.seed is not None:
        payload["input"]["seed"] = args.seed

    # ── Submit job ────────────────────────────────────────────────────────────
    url = f"{BASE_URL}/{args.endpoint}/run"
    print(f"Submitting job to endpoint {args.endpoint}…")
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"Job ID: {job_id}")

    # ── Poll for result ───────────────────────────────────────────────────────
    status_url = f"{BASE_URL}/{args.endpoint}/status/{job_id}"
    deadline   = time.time() + args.timeout
    delay      = 2

    while time.time() < deadline:
        time.sleep(delay)
        delay = min(delay * 1.4, 10)   # gentle back-off

        r = requests.get(status_url, headers=headers, timeout=30)
        r.raise_for_status()
        data   = r.json()
        status = data.get("status")
        print(f"  Status: {status}")

        if status == "COMPLETED":
            output = data.get("output", {})
            if "error" in output:
                sys.exit(f"Handler error: {output['error']}")

            img_b64 = output.get("image_base64", "")
            if not img_b64:
                sys.exit("No image in response.")

            with open(args.output, "wb") as f:
                f.write(base64.b64decode(img_b64))

            meta = output.get("meta", {})
            print(f"\n✅  Saved → {args.output}")
            print(f"   Generation time : {meta.get('generation_time_seconds')}s")
            print(f"   NSFW detected   : {output.get('nsfw_detected')}")
            print(f"   Meta            : {json.dumps(meta, indent=2)}")
            return

        if status in ("FAILED", "CANCELLED"):
            sys.exit(f"Job {status}: {data}")

    sys.exit(f"Timed out after {args.timeout}s waiting for job {job_id}")


if __name__ == "__main__":
    run(parse_args())
