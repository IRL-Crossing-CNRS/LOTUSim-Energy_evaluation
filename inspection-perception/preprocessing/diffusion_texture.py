"""
Low-strength diffusion img2img to inject realistic underwater texture/noise
onto a sim crack image, while preserving its geometry (crack shape/position).

Run this OUTSIDE the Claude sandbox — it needs to download a ~4-7GB model
from Hugging Face, which this environment's network doesn't allow.

Setup:
    pip install diffusers transformers accelerate torch --upgrade
    (use a CUDA build of torch if you have a GPU - strongly recommended,
     CPU inference will be very slow, minutes per image)

Usage:
    python diffusion_img2img_texture.py

Feed it your FDA-corrected sim image (color already nudged toward real) -
diffusion then adds texture/noise/blur on top. Doing both steps in sequence
(FDA first, diffusion second) tends to work better than diffusion alone,
since it starts closer to the target style already.
"""

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

MODEL_ID = "runwayml/stable-diffusion-v1-5"  # swap for a newer checkpoint if you have one
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None,  # optional: disable if it false-flags underwater/murky imagery
).to(DEVICE)

# Prompt describes the STYLE you want injected, not new content -
# keep it descriptive of texture/noise/lighting, not new objects
PROMPT = (
    "underwater concrete pillar with a crack, real underwater camera photo, "
    "murky turbid water, natural noise and blur, low contrast, "
    "biofouling and algae texture, photorealistic, raw underwater ROV footage"
)
NEGATIVE_PROMPT = (
    "clean, sharp, cartoon, render, cgi, video game, illustration, "
    "vector line, artificial, oversaturated, unrealistic"
)


def run_img2img(input_path, output_path, strength=0.2, guidance_scale=7.5, seed=0):
    """
    strength: how much the diffusion process is allowed to deviate from the
              input image. This is THE key parameter for your use case:
        0.1 - 0.2  -> subtle texture/noise added, geometry very safe
        0.3 - 0.4  -> more aggressive restyling, some risk of altering
                      the crack's shape/thinness
        0.5+       -> diffusion starts effectively regenerating the image;
                      don't go here if preserving exact crack geometry matters
        Start at 0.15, sweep upward only if the effect is too weak.
    """
    init_image = Image.open(input_path).convert("RGB")
    # SD1.5 works best around 512x512 - resize, generate, then upscale back
    orig_size = init_image.size
    init_image_resized = init_image.resize((512, 512))

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=init_image_resized,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    result = result.resize(orig_size)  # back to original resolution
    result.save(output_path)
    print(f"saved {output_path} (strength={strength})")


if __name__ == "__main__":
    # Sweep a few strengths so you can pick the one that keeps the crack
    # geometry intact while still shifting texture/noise meaningfully
    for s in [0.10, 0.15, 0.20, 0.30]:
        run_img2img(
            input_path="sim_crack_fda.jpg",   # your FDA-corrected sim crop
            output_path=f"sim_crack_diffusion_s{s}.jpg",
            strength=s,
        )