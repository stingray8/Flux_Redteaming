import torch
from diffusers import FluxPipeline
from Functions import *

MODEL_PATH = "/home/tingray/PycharmProjects/Flux In Painting/Models/FLUX.1-schnell/snapshots/schnell"

pipe = FluxPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)

cpu_use = True
if cpu_use:
    pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.to(torch.float16)


def create_image(seed, prompt, height, width, inferences, output_type='np'):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(
        prompt=prompt,
        guidance_scale=0.,
        height=height,
        width=width,
        num_inference_steps=inferences,
        max_sequence_length=32,
        generator=generator,  # pass the generator here
        output_type=output_type
    ).images[0]
    return out


np_to_img(create_image(0, "face", 512, 512, 10), "face.png")
