import pytest
import torch

from shared.config import (
    Settings,
    load_settings,
    resolve_device,
    resolve_dtype,
    seed_everything,
)


def test_default_settings_from_config_file():
    s = load_settings()
    assert s.device == "auto"
    assert s.seed == 0
    assert s.dtype == "float32"


def test_resolve_device_returns_torch_device():
    assert isinstance(resolve_device(Settings(device="auto")), torch.device)
    assert resolve_device(Settings(device="cpu")).type == "cpu"


def test_resolve_dtype():
    assert resolve_dtype(Settings(dtype="float32")) is torch.float32
    assert resolve_dtype(Settings(dtype="float64")) is torch.float64


def test_env_override_device(monkeypatch):
    monkeypatch.setenv("AIEM_DEVICE", "cpu")
    assert load_settings().device == "cpu"


def test_invalid_device_raises(monkeypatch):
    monkeypatch.setenv("AIEM_DEVICE", "gpu")
    with pytest.raises(ValueError):
        load_settings()


def test_invalid_dtype_raises(monkeypatch):
    monkeypatch.setenv("AIEM_DTYPE", "float8")
    with pytest.raises(ValueError):
        load_settings()


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_env_override_seed(monkeypatch):
    monkeypatch.setenv("AIEM_SEED", "7")
    assert load_settings().seed == 7


def test_invalid_seed_raises(monkeypatch):
    monkeypatch.setenv("AIEM_SEED", "not-an-int")
    with pytest.raises(ValueError):
        load_settings()


def test_configure_returns_device_and_is_reproducible(monkeypatch):
    monkeypatch.setenv("AIEM_DEVICE", "cpu")
    from shared.config import configure

    device = configure()
    assert device.type == "cpu"
    a = torch.randn(4)
    configure()
    b = torch.randn(4)
    assert torch.equal(a, b)
