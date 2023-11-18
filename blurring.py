from PIL import ImageFilter

from utils import compose_two_images_according_to_mask


def blur(image, mask):
    """
    image: <PIL.Image>
    mask: <PIL.Image>

    return: <PIL.Image>

    This function blurs the masked region.
    """

    blured_image = image.filter(ImageFilter.GaussianBlur(5))
    output_image = compose_two_images_according_to_mask(
        base_image=image, image_for_masked_region=blured_image, mask=mask
    )
    return output_image
