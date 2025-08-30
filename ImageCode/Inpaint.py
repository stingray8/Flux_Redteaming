import torch
from diffusers import FluxFillPipeline
from Functions import *

MODEL_PATH = "/home/tingray/PycharmProjects/Flux In Painting/Models/FLUX.1-Fill-dev/snapshots/Fill-dev"

pipe = FluxFillPipeline.from_pretrained(
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


def inpaint_image(seed, prompt, image, mask, height, width, inferences, output_type='np'):
    """
    Perform image inpainting using a diffusion pipeline.
    3d image
    2d mask (0 is cover, 255 is place to be in-painted

    Parameters
    ----------
    seed : int
        Random seed for reproducibility. Used by the torch.Generator.

    prompt : str
        Text prompt that guides the inpainting.

    image : PIL.Image.Image or numpy.ndarray
        The base image to be inpainted.
        - If numpy.ndarray, it will be converted to a PIL image internally.

    mask : PIL.Image.Image or numpy.ndarray
        The binary mask indicating regions for inpainting.
        - Mask should be 2D.
        - Convention: `0` = covered region (to keep), `255` = region to inpaint.
        - If numpy.ndarray, it will be converted to a PIL image internally.

    height : int
        Desired output image height in pixels.

    width : int
        Desired output image width in pixels.

    inferences : int
        Number of inference steps for the diffusion model (higher = higher quality, slower).

    output_type : str, optional (default = 'np')
        Type of output image.
        - `'np'` → returns a `numpy.ndarray`
        - `'pil'` → returns a `PIL.Image.Image`
        - `'pt'` → returns a `torch.Tensor` (if supported by pipeline)

    Returns
    -------
    out : numpy.ndarray or PIL.Image.Image
        The inpainted image, format depending on `output_type`.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)

    if not isinstance(image, Image.Image):
        assert isinstance(image, np.ndarray)
        image = np_to_pil(image)

    if not isinstance(mask, Image.Image):
        assert isinstance(mask, np.ndarray)
        mask = np_to_pil(mask)

    out = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=height,
        width=width,
        num_inference_steps=inferences,
        max_sequence_length=32,
        generator=generator,
        output_type=output_type,
        # output_type="pt" # Tensor
    ).images[0]
    return out
