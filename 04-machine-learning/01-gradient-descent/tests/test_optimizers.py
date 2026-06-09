import torch

from gd.models import LinearModel
from gd.optimizers import SGD, Momentum, RMSProp, Adam


def _problem():
    torch.manual_seed(0)
    X = torch.randn(64, 2)
    true_w = torch.tensor([2.0, -1.0])
    y = X @ true_w + 0.5
    return X, y


def test_sgd_reduces_loss():
    X, y = _problem()
    model = LinearModel(w=torch.zeros(2), b=torch.zeros(()))
    opt = SGD(lr=0.1)
    start = model.loss(X, y).item()
    for _ in range(200):
        gw, gb = model.gradients(X, y)
        new_w, new_b = opt.step([model.w, model.b], [gw, gb])
        model = LinearModel(w=new_w, b=new_b)
    assert model.loss(X, y).item() < start * 0.1


def test_scratch_adam_matches_torch_optim_adam():
    X, y = _problem()
    # from-scratch
    model = LinearModel(w=torch.zeros(2), b=torch.zeros(()))
    opt = Adam(lr=0.05)
    for _ in range(300):
        gw, gb = model.gradients(X, y)
        nw, nb = opt.step([model.w, model.b], [gw, gb])
        model = LinearModel(w=nw, b=nb)

    # torch reference
    w = torch.zeros(2, requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    ref = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(300):
        ref.zero_grad()
        loss = torch.mean((X @ w + b - y) ** 2)
        loss.backward()
        ref.step()

    assert torch.allclose(model.w, w.detach(), atol=1e-2)
    assert torch.allclose(model.b, b.detach(), atol=1e-2)


def test_momentum_reduces_loss():
    X, y = _problem()
    model = LinearModel(w=torch.zeros(2), b=torch.zeros(()))
    opt = Momentum(lr=0.1)
    start = model.loss(X, y).item()
    for _ in range(200):
        gw, gb = model.gradients(X, y)
        nw, nb = opt.step([model.w, model.b], [gw, gb])
        model = LinearModel(w=nw, b=nb)
    assert model.loss(X, y).item() < start * 0.1


def test_rmsprop_reduces_loss():
    X, y = _problem()
    model = LinearModel(w=torch.zeros(2), b=torch.zeros(()))
    opt = RMSProp(lr=0.1)
    start = model.loss(X, y).item()
    for _ in range(500):
        gw, gb = model.gradients(X, y)
        nw, nb = opt.step([model.w, model.b], [gw, gb])
        model = LinearModel(w=nw, b=nb)
    assert model.loss(X, y).item() < start * 0.1
