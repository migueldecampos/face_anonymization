import numpy as np

from PIL import Image, ImageDraw
from insightface.app import FaceAnalysis

FACEAN = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
FACEAN.prepare(ctx_id=0, det_size=(640, 640))


def get_faces_from_image(image):
    """
    image: <PIL.Image>

    return: <list: PIL.Image>

    This function receives an image, and returns a list of masks corresponding to the faces
    found in the image.
    """

    faces = FACEAN.get(np.array(image))

    masks = list()
    for face in faces:
        box = face.bbox.astype(int)
        top_left_corner = (box[0], box[1])
        bottom_right_corner = (box[2], box[3])
        radius = min(box[2] - box[0], box[3] - box[1]) // 3
        mask = Image.new("RGB", (image.width, image.height), "black")
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle(
            [top_left_corner, bottom_right_corner],
            radius=radius,
            fill="white",
            corners=(True, True, True, True),
        )
        masks.append((mask, top_left_corner, bottom_right_corner))

    return masks


if __name__ == "__main__":
    image = Image.open("examples/friends.jpg")

    masks = get_faces_from_image(image)

    print(len(masks))
    for i, mask in enumerate(masks):
        mask.save("examples/friends_mask{}.jpg".format(i))
