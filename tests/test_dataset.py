# Sample Test passing with nose and pytest
from torchvision.datasets import VOCDetection
import yaml


def test_params_exists():
    with open("params.yaml") as file:
        params = yaml.safe_load(file)
    assert len(params) > 0


def test_dataset():
    with open("params.yaml") as file:
        params = yaml.safe_load(file)
    params["classes"] = set(params["classes"])
    voc_dataset = VOCDetection(params["locations"]["data"])
    assert len(voc_dataset) > 0
