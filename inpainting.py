import torch
from diffusers import StableDiffusionInpaintPipeline

from utils import get_device, compose_two_images_according_to_mask

PIPELINE = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
PIPELINE.to(get_device())

MODEL_WIDTH = 512
MODEL_HEIGHT = 512


def _get_region(width, height, mask_top_left_corner, mask_bottom_right_corner):
    x0, y0 = mask_top_left_corner
    x1, y1 = mask_bottom_right_corner

    # Calculate width and height of the rectangle
    rect_width = x1 - x0
    rect_height = y1 - y0

    # Calculate center of the rectangle
    center_x = x0 + (rect_width / 2)
    center_y = y0 + (rect_height / 2)

    if center_x - MODEL_WIDTH / 2 < 0:
        crop_left = 0
        crop_right = 0 + MODEL_WIDTH
    elif center_x + MODEL_WIDTH / 2 > width:
        crop_right = width
        crop_left = width - MODEL_WIDTH
    else:
        crop_left = int(center_x - MODEL_WIDTH / 2)
        crop_right = crop_left + MODEL_WIDTH

    if center_y - MODEL_HEIGHT / 2 < 0:
        crop_upper = 0
        crop_lower = crop_upper + MODEL_HEIGHT
    elif center_y + MODEL_HEIGHT / 2 > height:
        crop_lower = height
        crop_upper = height - MODEL_HEIGHT
    else:
        crop_upper = int(center_y - MODEL_HEIGHT / 2)
        crop_lower = crop_upper + MODEL_HEIGHT

    region_top_left_corner = crop_upper, crop_left
    region_bottom_right_corner = crop_lower, crop_right
    return region_top_left_corner, region_bottom_right_corner


def inpaint(image, mask, prompt=""):
    """
    image: <PIL.Image>
    mask: (<PIL.Image>, (int, int), (int, int)>
    prompt: <str>

    return: <PIL.Image>

    This function uses PIPELINE to inpaint the area under _mask_.
    """

    mask_image = mask[0]
    original_image = image.copy()
    region_top_left_corner, region_bottom_right_corner = _get_region(
        width=image.width,
        height=image.height,
        mask_top_left_corner=mask[1],
        mask_bottom_right_corner=mask[2],
    )
    smaller_image = image.crop(
        (
            region_top_left_corner[1],
            region_top_left_corner[0],
            region_bottom_right_corner[1],
            region_bottom_right_corner[0],
        )
    )
    smaller_mask = mask_image.crop(
        (
            region_top_left_corner[1],
            region_top_left_corner[0],
            region_bottom_right_corner[1],
            region_bottom_right_corner[0],
        )
    )
    inpainted_image = PIPELINE(
        prompt=prompt,
        image=smaller_image,
        mask_image=smaller_mask,
    ).images[0]

    image.paste(
        inpainted_image,
        (
            region_top_left_corner[1],
            region_top_left_corner[0],
        ),
    )

    output_image = compose_two_images_according_to_mask(
        base_image=original_image, image_for_masked_region=image, mask=mask_image
    )

    return output_image


if __name__ == "__main__":
    import sys

    # import cv2
    from PIL import Image

    image = Image.open("images/small_friends.jpg")
    mask = Image.open("images/small_friends_mask.jpg")

    inpainted_image = inpaint(image, mask, prompt=" ".join(sys.argv[2:]))
    inpainted_image.save("images/small_friends_output_{}.png".format(sys.argv[1]))
