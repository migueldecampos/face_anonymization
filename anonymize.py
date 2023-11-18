import functools

from face_detection import get_faces_from_image
from inpainting import inpaint
from blurring import blur


def anonymize_faces(image, filter):
    """
    image: <PIL.Image>
    filter: <function: image, mask -> image>

    return: <PIL.Image>

    This function anonymizes faces using _filter_.
    """

    face_masks = get_faces_from_image(image)

    for mask in face_masks:
        image = filter(image, mask)

    return image


blur_faces = functools.partial(anonymize_faces, filter=blur)
replace_faces = functools.partial(anonymize_faces, filter=inpaint)


if __name__ == "__main__":
    from PIL import Image

    image = Image.open("images/big_friends.jpg")

    anonymized_image = replace_faces(image)
    anonymized_image.save("images/big_friends_output.jpg")
