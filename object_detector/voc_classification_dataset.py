from torchvision.datasets import VOCDetection
from pathlib import Path
from typing import Dict, Tuple
import PIL


class VOCClassification:
    def __init__(self, data: Path, classes: Dict, transforms) -> None:
        """A image classifier class based on the VOC Detection
        dataset. SImply because I wanted to see if I could change
        between these two contexts. This class just takes the first
        item from the list of annotations

        Parameters
        ----------
        data : Path
            [description]
        classes : Dict
            A dictionary of string (name) to class (int)
        """
        self.classes = classes
        self.dataset = VOCDetection(data)
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def _calculate_class(self, name):
        return self.classes[name]

    def bndbox_to_tuple(
        self, bndbox: Dict[str, str]
    ) -> Tuple[float, float, float, float]:
        """To comply with PIL draw rectangle function
        https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html#PIL.ImageDraw.ImageDraw.rectangle
        """
        return (
            float(bndbox["xmin"]),
            float(bndbox["ymin"]),
            float(bndbox["xmax"]),
            float(bndbox["ymax"]),
        )

    def _crop_image(self, image: PIL.Image.Image, bndbox):
        return image.crop((self.bndbox_to_tuple(bndbox)))

    def __getitem__(self, idx):
        image, annotation = self.dataset[idx]
        annotation = annotation["annotation"]["object"][0]
        image = self._crop_image(image, annotation["bndbox"])
        classification = self._calculate_class(annotation["name"])
        image, classification = self.transforms(image, classification)
        return image, classification


def main():
    classes = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    classes_dict = {}
    for idx in range(0, len(classes)):
        classes_dict[classes[idx]] = idx

    dataset = VOCClassification(Path("data"), classes_dict)
    test_data = dataset[8]
    test_data[0].show()
    print(test_data[1])


if __name__ == "__main__":
    main()
