import torch
from diffusers import StableDiffusionInpaintPipeline

from utils import get_device, compose_two_images_according_to_mask

PIPELINE = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
PIPELINE.to(get_device())


def inpaint(image, mask, prompt=""):
    """
    image: <PIL.Image>
    mask: <PIL.Image>
    prompt: <str>

    return: <PIL.Image>

    This function uses PIPELINE to inpaint the area under _mask_.
    """

    inpainted_image = PIPELINE(
        prompt=prompt,
        image=image,
        mask_image=mask,
    ).images[0]

    output_image = compose_two_images_according_to_mask(
        base_image=image, image_for_masked_region=inpainted_image, mask=mask
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
