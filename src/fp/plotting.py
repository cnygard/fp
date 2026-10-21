"""Optional live or saved reconstruction views driven by detached state snapshots."""

from math import ceil, pi
from pathlib import Path

import torch

from .fft import ifft2c
from .results import to_numpy

DEFAULT_VIEWS = ("amplitude", "spectrum", "pupil_amplitude", "pupil_phase", "residual", "convergence")
AVAILABLE_VIEWS = DEFAULT_VIEWS + ("phase", "measurement", "prediction", "update")


class ReconstructionVisualizer:
    """State callback with selectable panels; custom callbacks may use any extras.

    show=False with output_dir writes PNG snapshots without a GUI. Closing the
    live window stops rendering. Core solvers neither import pyplot nor plot.
    """

    def __init__(self, *, views=DEFAULT_VIEWS, show=True, output_dir=None):
        self.views = tuple(views)
        if not self.views or any(view not in AVAILABLE_VIEWS for view in self.views):
            raise ValueError(f"views must be selected from {AVAILABLE_VIEWS}")
        self.show = show
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        rows, columns = ceil(len(self.views) / 3), min(3, len(self.views))
        self.plt = None
        if show:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.figure, axes = plt.subplots(rows, columns, figsize=(13, 4 * rows), squeeze=False)
        else:
            # Agg renders directly to files without initializing a GUI backend.
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            self.figure = Figure(figsize=(13, 4 * rows))
            FigureCanvasAgg(self.figure)
            axes = self.figure.subplots(rows, columns, squeeze=False)
        self.axes = list(axes.flat)
        self.metrics = {}
        self.disabled = False
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        if show:
            self.plt.show(block=False)

    def _on_close(self, event):
        self.disabled = True

    def __call__(self, state):
        if self.disabled:
            return
        if state.metrics is not None:
            self.metrics[state.metrics.epoch] = state.metrics.amplitude_rmse
        object_field = None
        if "amplitude" in self.views or "phase" in self.views:
            object_field = ifft2c(state.spectrum)
        for axis in self.axes:
            axis.clear()
            axis.set_visible(False)
        for axis, view in zip(self.axes, self.views):
            axis.set_visible(True)
            if view == "convergence":
                epochs = sorted(self.metrics)
                axis.plot(epochs, [self.metrics[e] for e in epochs], marker=".")
                axis.set_title("Amplitude RMSE")
                axis.set_xlabel("Epoch")
                axis.grid(alpha=0.2)
                continue
            cmap, limits = "gray", {}
            title = view.replace("_", " ").title()
            if view == "amplitude":
                image = object_field.abs()
            elif view == "phase":
                image, cmap, limits = object_field.angle(), "twilight", {"vmin": -pi, "vmax": pi}
            elif view == "spectrum":
                image, title = state.spectrum.abs().log1p(), "Object spectrum: log(1 + magnitude)"
            elif view == "pupil_amplitude":
                image = state.pupil.abs()
            elif view == "pupil_phase":
                image = torch.where(state.pupil.abs() > 0, state.pupil.angle(), torch.nan)
                cmap, limits = "twilight", {"vmin": -pi, "vmax": pi}
            else:
                image = state.extras.get(view)
                if image is None:
                    axis.text(0.5, 0.5, "No frame yet", ha="center", va="center", transform=axis.transAxes)
                    axis.set_title(title)
                    axis.axis("off")
                    continue
                if view == "update":
                    image, title = image.abs().log1p(), "Patch update: log(1 + magnitude)"
                elif view == "residual":
                    bound = max(float(image.abs().max().item()), 1e-12)
                    cmap, limits = "coolwarm", {"vmin": -bound, "vmax": bound}
            axis.imshow(to_numpy(image), cmap=cmap, **limits)
            axis.set_title(title)
            axis.axis("off")
        frame = "" if state.frame_index is None else f", frame {state.frame_index}"
        self.figure.suptitle(f"Epoch {state.epoch}, update {state.step}{frame}")
        self.figure.tight_layout()
        if self.output_dir is not None:
            self.figure.savefig(self.output_dir / f"step_{state.step:07d}.png", dpi=120)
        if self.show:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            self.plt.pause(0.001)

    def finish(self):
        if self.show and not self.disabled:
            self.plt.show(block=True)

    def close(self):
        self.disabled = True
        if self.plt is not None:
            self.plt.close(self.figure)
        else:
            self.figure.clear()


def plot_result(result):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(to_numpy(result.amplitude), cmap="gray")
    axes[0].set_title("Amplitude")
    axes[1].imshow(to_numpy(result.phase), cmap="twilight", vmin=-pi, vmax=pi)
    axes[1].set_title("Phase (radians)")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    plt.show()
