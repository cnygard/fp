"""Optional presentation; importing the numerical package does not import pyplot."""


def plot_result(result):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(result.amplitude, cmap="gray")
    axes[0].set_title("Amplitude")
    axes[1].imshow(result.phase, cmap="twilight", vmin=-3.141592653589793, vmax=3.141592653589793)
    axes[1].set_title("Phase (radians)")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    plt.show()
