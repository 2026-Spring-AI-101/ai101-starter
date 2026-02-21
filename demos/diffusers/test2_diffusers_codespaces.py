#!/usr/bin/env python3
"""
Codespaces-friendly Diffusers image generation (CPU).

Why this version?
- Removes hard-coded local Windows/WSL paths.
- Uses HF Hub model IDs with an explicit cache_dir inside the repo.
- Saves images to ./outputs/ (no GUI popup needed).
- Provides CPU-friendly defaults (Turbo / distilled models, few steps).

Recommended models for CPU classroom demos:
- stabilityai/sd-turbo  (1-4 steps sampling)  (fastest general choice)
- segmind/tiny-sd       (distilled SD1.5 UNet, smaller/faster than full SD1.5)

Usage examples:
  python test2_diffusers_codespaces.py --model stabilityai/sd-turbo --prompt "a cute robot" --steps 2
  python test2_diffusers_codespaces.py --model segmind/tiny-sd --prompt "a watercolor fox reading a book" --steps 8
  python test2_diffusers_codespaces.py --model stable-diffusion-v1-5/stable-diffusion-v1-5 --steps 20

Notes:
- Some models may require you to accept a license on Hugging Face and/or login with a token.
  In Codespaces you can do:  huggingface-cli login
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("DIFFUSERS_MODEL", "stabilityai/sd-turbo"),
                  help="HF model id or local path. Recommended: stabilityai/sd-turbo or segmind/tiny-sd")
    p.add_argument("--prompt", default="vampire princess fall into love with mummy prince",
                  help="Text prompt.")
    p.add_argument("--negative", default="",
                  help="Negative prompt (optional).")
    p.add_argument("--steps", type=int, default=int(os.environ.get("DIFFUSERS_STEPS", "2")),
                  help="Inference steps. sd-turbo: 1-4. tiny-sd: ~6-12. SD1.5 full: 20-30.")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/output.png",
                  help="Output image path.")
    p.add_argument("--cache-dir", default=os.environ.get("HF_CACHE_DIR", "models/diffusers"),
                  help="Where to cache HF models inside the repo.")
    p.add_argument("--threads", type=int, default=0,
                  help="CPU threads (0 = let PyTorch decide).")
    p.add_argument("--guidance-scale", type=float, default=None,
                  help="CFG guidance scale. For Turbo models it's typically 0.0; leave empty to auto-set.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # CPU-friendly dtype: float32 on CPU
    torch_dtype = torch.float32

    # Auto-set guidance defaults:
    # - sd-turbo / sdxl-turbo typically use guidance_scale=0.0
    # - classic SD models often use 7.5
    if args.guidance_scale is None:
        if "turbo" in args.model.lower():
            guidance_scale = 0.0
        else:
            guidance_scale = 7.5
    else:
        guidance_scale = args.guidance_scale

    print("Model:", args.model)
    print("Cache dir:", cache_dir.resolve())
    print("Prompt:", args.prompt)
    print("Steps:", args.steps, "Guidance:", guidance_scale)
    print("Size:", args.width, "x", args.height)

    # Load pipeline
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        cache_dir=str(cache_dir),
        local_files_only=False,
        safety_checker=None,  # keep simple for class; you can re-enable if needed
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")

    # CPU memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    # Deterministic seed
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    print("Generating image... (CPU may take a while the first time due to model download)")
    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            negative_prompt=(args.negative if args.negative.strip() else None),
            num_inference_steps=args.steps,
            guidance_scale=guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images[0]

    image.save(out_path)
    print("Saved ->", out_path.resolve())
    print("Tip: in Codespaces, open the file in the Explorer to preview it.")


if __name__ == "__main__":
    main()
