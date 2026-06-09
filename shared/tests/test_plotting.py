from matplotlib.axes import Axes
from shared.plotting import plot_loss_surface, plot_descent_path


def test_plot_loss_surface_returns_axes():
    def loss(w, b):
        return (w - 1.0) ** 2 + (b + 2.0) ** 2

    ax = plot_loss_surface(loss, w_range=(-3, 3), b_range=(-3, 3), steps=20)
    assert isinstance(ax, Axes)


def test_plot_descent_path_returns_axes_and_keeps_points():
    path = [(0.0, 0.0), (0.5, -0.5), (0.9, -1.5)]
    ax = plot_descent_path(path)
    assert isinstance(ax, Axes)
    line = ax.get_lines()[0]
    assert len(line.get_xdata()) == 3
