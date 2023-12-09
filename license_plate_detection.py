import torch
import torchvision
from PIL import Image, ImageDraw


SCORE_CUTOFF = 0.7


def get_license_plate_boxes(model, image):
    """
    image: <torch.Tensor>, as out of torchvision.io.read_image(image_path)

    return: <torch.Tensor> of shape (N, 4), where N is the number of license plates found.
    """

    image_norm = image / 255
    model.eval()
    x = [image_norm]
    predictions = model(x)  # Returns predictions
    predictions = predictions[0]

    boxes = list()
    for box, label, score in zip(
        predictions["boxes"], predictions["labels"], predictions["scores"]
    ):
        if label == 1:
            if score >= SCORE_CUTOFF:
                boxes.append(box)
            else:
                break

    return torch.stack(boxes) if boxes else torch.zeros((0, 4))


def get_license_plates_from_image(model, image_path):
    """
    image: <torch.Tensor>, as out of torchvision.io.read_image(image_path)

    return: <list: PIL.Image>

    This function receives an image, and returns a list of masks corresponding to the
    license plates found in the image.
    """

    license_plate_boxes = get_license_plate_boxes(
        model, torchvision.io.read_image(image_path)
    )

    image = Image.open(image_path)

    masks = list()
    i = 0
    for box in license_plate_boxes:
        box = box.int()
        top_left_corner = (box[0], box[1])
        bottom_right_corner = (box[2], box[3])
        mask = Image.new("RGB", (image.width, image.height), "black")
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle(
            [top_left_corner, bottom_right_corner],
            radius=0,
            fill="white",
            corners=(True, True, True, True),
        )
        masks.append((mask, top_left_corner, bottom_right_corner))
        i = i + 1

    return masks
