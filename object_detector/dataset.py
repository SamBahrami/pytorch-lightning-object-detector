from torchvision.datasets import VOCDetection, voc
import yaml
import PIL
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple


def bndbox_to_tuple(bndbox: Dict[str, str]) -> Tuple[float, float, float, float]:
    """To comply with PIL draw rectangle function
    https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html#PIL.ImageDraw.ImageDraw.rectangle
    """
    return (
        float(bndbox["xmin"]),
        float(bndbox["ymin"]),
        float(bndbox["xmax"]),
        float(bndbox["ymax"]),
    )


def string_to_number(s) -> int:
    """A way to convert a string into an int
    little means little endianness
    https://stackoverflow.com/questions/31701991/string-of-text-to-unique-integer-method

    Parameters
    ----------
    s : string
        string to encode

    Returns
    -------
    int
        a unique replicable number for the string s
    """
    return int.from_bytes(s.encode(), "little")


def draw_bounding_boxes(
    image: PIL.Image.Image, voc_object_annotations: List[Dict], params: Dict
) -> PIL.Image.Image:
    box_image = image.copy()
    draw = ImageDraw.Draw(box_image)
    for annotation in voc_object_annotations:
        bndbox = annotation["bndbox"]
        tuple_bndbox = bndbox_to_tuple(bndbox)
        box_colour = string_to_number(annotation["name"]) % 0xFFFFFF
        box_colour = params["colours"][
            string_to_number(annotation["name"]) % len(params["colours"])
        ]
        draw.rectangle(tuple_bndbox, outline=box_colour)
        draw.text(tuple_bndbox[:2], annotation["name"], fill=box_colour)
    return box_image


def generate_list_of_classes(voc_dataset):
    """Couldn't find a good list of classes online
    so I thought I could just generate it by iterating through
    the dataset

    Parameters
    ----------
    voc_dataset : pytorch dataset
        voc dataset from pytorch torchvision datasets (2012)
    """
    classes = set()
    for i in range(2000):
        _, test_annotation = voc_dataset[i]
        for annotation in test_annotation["annotation"]["object"]:
            classes.add(annotation["name"])
    print(classes)


def main():
    """Make sure the dataset works as intended. Example loading
    and previewing images from the dataset.
    """
    with open("params.yaml") as file:
        params = yaml.safe_load(file)
    params["classes"] = set(params["classes"])
    voc_dataset = VOCDetection(params["locations"]["data"])
    test_data = voc_dataset[3]
    test_image = draw_bounding_boxes(
        test_data[0], test_data[1]["annotation"]["object"], params
    )
    test_image.show()
    return


if __name__ == "__main__":
    main()
