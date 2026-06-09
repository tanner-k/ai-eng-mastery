import torch
from shared.devices import get_device


def test_get_device_returns_torch_device():
    assert isinstance(get_device(), torch.device)


def test_get_device_respects_explicit_cpu():
    assert get_device("cpu").type == "cpu"


def test_get_device_rejects_invalid_prefer():
    import pytest

    with pytest.raises(ValueError):
        get_device("gpu")


def test_priority_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device().type == "cuda"


def test_priority_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert get_device().type == "mps"


def test_priority_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device().type == "cpu"
