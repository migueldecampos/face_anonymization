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
    import sys
    from PIL import Image

    if len(sys.argv) == 3 and sys.argv[1].lower() in ("blur", "inpaint"):
        image = Image.open(sys.argv[2])
        if sys.argv[1].lower() == "inpaint":
            anonymized_image = replace_faces(image)
        else:
            anonymized_image = blur_faces(image)

        anonymized_image.save(sys.argv[2][:-4] + "_{}.jpg".format(sys.argv[1].lower()))
    else:
        print("python anonymize.py blur|inpaint <path to your image>")
