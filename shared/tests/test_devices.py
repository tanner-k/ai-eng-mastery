import torch
from shared.devices import get_device


def test_get_device_returns_torch_device():
    assert isinstance(get_device(), torch.device)


def test_get_device_respects_explicit_cpu():
    assert get_device("cpu").type == "cpu"
