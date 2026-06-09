import torch

from gd.models import LinearModel


def test_manual_gradients_match_autograd():
    torch.manual_seed(0)
    X = torch.randn(32, 3)
    true_w = torch.tensor([1.0, -2.0, 0.5])
    y = X @ true_w + 0.3

    w = torch.zeros(3, requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    model = LinearModel(w=w, b=b)
    loss = model.loss(X, y)
    loss.backward()

    grad_w, grad_b = LinearModel(w=w.detach(), b=b.detach()).gradients(X, y)
    assert torch.allclose(grad_w, w.grad, atol=1e-5)
    assert torch.allclose(grad_b, b.grad, atol=1e-5)
