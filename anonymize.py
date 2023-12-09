import os
from PIL import Image


def anonymize_faces(image, filter):
    """
    image: <PIL.Image>
    filter: <function: image, mask -> image>

    return: <PIL.Image>

    This function anonymizes faces using _filter_.
    """
    from face_detection import get_faces_from_image

    face_masks = get_faces_from_image(image)

    for mask in face_masks:
        image = filter(image, mask)

    return image


def blur_faces(image):
    from blurring import blur

    return anonymize_faces(image, blur)


def replace_faces(image):
    from inpainting import inpaint

    return anonymize_faces(image, inpaint)


def anonymize_license_plates(model, image_path):
    """
    model: torch model from https://github.com/migueldecampos/license_plate_detection
    image: <string>

    return: <PIL.Image>

    This function blurs license plates.
    """
    from license_plate_detection import get_license_plates_from_image
    from blurring import blur

    license_plate_masks = get_license_plates_from_image(
        model, os.path.abspath(image_path)
    )
    image = Image.open(image_path)
    for mask in license_plate_masks:
        image = blur(image, mask)

    return image


if __name__ == "__main__":
    import pickle
    import sys

    FACES_COMMAND = "python anonymize.py faces blur|inpaint <path to your image>"
    LICENSE_PLATES_COMMAND = "python anonymize.py license-plates <path to model pickle file> <path to your image>"

    if len(sys.argv) == 4 and sys.argv[1].lower() in ("faces", "license-plates"):
        if sys.argv[1].lower() == "faces":
            mode = sys.argv[2].lower()
            if mode in ("blur", "inpaint"):
                image = Image.open(sys.argv[3])
                if mode == "inpaint":
                    anonymized_image = replace_faces(image)
                else:
                    anonymized_image = blur_faces(image)
            else:
                print(FACES_COMMAND)
        else:
            mode = "blur"
            with open(sys.argv[2], "rb") as p:
                model = pickle.load(p)
            anonymized_image = anonymize_license_plates(
                model=model, image_path=sys.argv[3]
            )
        anonymized_image.save(sys.argv[3][:-4] + "_{}.jpg".format(mode))
    else:
        print(FACES_COMMAND, "or", LICENSE_PLATES_COMMAND, sep="\n")
