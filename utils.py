import numpy as np
import torch
from PIL import Image


def get_device():
    return (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def compose_two_images_according_to_mask(base_image, image_for_masked_region, mask):
    """
    base_image: <PIL.Image>
    image_for_masked_region: <PIL.Image>
    mask: <PIL.Image>

    return: <PIL.Image>

    This function takes a mask (containing black and white), a two images.
    It will return a composite of the two images, containing pixels from base_image
    for the mask's black region, and from image_for_masked_region for the white region.
    """

    mask_arr = np.array(mask.convert("L"))
    mask_arr = mask_arr[:, :, None]
    # Binarize the mask: 1s correspond to the pixels which are repainted
    mask_arr = mask_arr.astype(np.float32) / 255.0
    mask_arr[mask_arr < 0.5] = 0
    mask_arr[mask_arr >= 0.5] = 1

    output_image = (1 - mask_arr) * base_image + mask_arr * image_for_masked_region
    output_image = Image.fromarray(output_image.round().astype("uint8"))
    return output_image
