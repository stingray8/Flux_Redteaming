from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert a NumPy array to a PIL Image.

    Parameters:
        arr (np.ndarray): Input array. Can be:
            - RGB: shape [H, W, 3], float in [0,1] or uint8 [0,255]
            - Grayscale: shape [H, W], float in [0,1] or uint8 [0,255]

    Returns:
        PIL.Image.Image
    """
    # Convert float [0,1] -> uint8 [0,255]
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0, 1)  # ensure within [0,1]
        arr = (arr * 255).astype(np.uint8)
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.uint8)
    else:
        raise TypeError(f"Unsupported array dtype: {arr.dtype}")

    # Decide mode
    if arr.ndim == 2:
        mode = "L"  # grayscale
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    return Image.fromarray(arr, mode=mode)


def pil_to_np(pil: Image.Image) -> np.ndarray:
    return np.array(pil)


def pil_to_img(pil: Image.Image, file: str):
    pil.save(file)


def np_to_img(arr: np.ndarray, file: str):
    pil_to_img(np_to_pil(arr), file)


def img_to_np(file: str):
    return np.array(Image.open(file))


def img_to_pil(img_file: str, img_size=512):
    """
    Loads an image file and converts it to a PIL Image.

    Args:
        img_file (str or Path): Path to the image file.

    Returns:
        PIL.Image.Image: The loaded image.
        :param img_size: pixels x pixels
    """
    img = Image.open(img_file).convert("RGB")  # ensures 3 channels
    img = img.resize((img_size, img_size), Image.LANCZOS)
    return img

def get_abs_path(*path_parts):
    """
    Returns the absolute path to a file, starting from the parent folder of the current script.

    Args:
        *path_parts: Any number of strings representing folder names or the file name.

    Returns:
        str: Absolute path to the target file.
    """
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go to parent folder
    parent_dir = os.path.dirname(current_dir)

    # Join the parent folder with the provided path parts
    full_path = os.path.join(parent_dir, *path_parts)

    # Return the absolute path
    return os.path.abspath(full_path)


def show_overlay_mask(image_np, mask_np, color=(255, 0, 0), alpha=0.4):
    """
    Overlay a binary/continuous mask on top of an RGB image.

    Args:
        image_np (np.ndarray): HxWx3 uint8 array (RGB image)
        mask_np  (np.ndarray): HxW array (binary or continuous mask)
        color (tuple): Overlay color (R, G, B)
        alpha (float): Transparency of overlay [0..1]

    Returns:
        np.ndarray: HxWx3 uint8 array with overlay applied
    """
    # Ensure RGB
    if image_np.ndim == 2:  # grayscale -> RGB
        image_np = np.stack([image_np] * 3, axis=-1)

    # Normalize mask to [0,1]
    mask_norm = (mask_np > 0).astype(np.float32)

    # Apply overlay
    overlay = image_np.copy().astype(np.float32)
    for c in range(3):
        overlay[..., c] = (
                image_np[..., c] * (1 - alpha * mask_norm) +
                color[c] * (alpha * mask_norm)
        )

    plt.imshow(overlay.astype(np.uint8))
    plt.axis("off")
    plt.show()
