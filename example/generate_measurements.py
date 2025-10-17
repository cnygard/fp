import torch
import numpy as np
from forward_model import *
from config import *
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import cv2


if __name__ == "__main__":
    # === Load and preprocess image ===
    sample = cv2.imread("/Users/m/Downloads/usaf.jpg", 0).astype(np.float32)
    sample = sample / 255.0  # normalize
    sample = torch.from_numpy(sample)

    config = setup_config(sample, base_config=FPM_CONFIG)

    theta_max = np.rad2deg(np.arcsin(config["NA_illum"]))
    print(f"Max rotation (deg) {theta_max:.2e}")

    I_center = forward_model(sample, torch.tensor([0, 0]), config)
    I_shift = forward_model(sample, torch.tensor([0, theta_max / 4]), config)

    # === Display ===
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Config: {config['title']}")

    # Ground truth sample amplitude
    im00 = axs[0, 0].imshow(sample, cmap="gray")
    axs[0, 0].set_title(f"Sample (z={config['object_distance']:.2e}m)")
    fig.colorbar(im00, ax=axs[0, 0])
    scalebar = ScaleBar(config["dx"], "m", location="lower right")
    axs[0, 0].add_artist(scalebar)

    im01 = axs[0, 1].imshow(config["M"])
    axs[0, 1].set_title("Pupil mask")
    fig.colorbar(im01, ax=axs[0, 1])
    scalebar = ScaleBar(config["dx"], "m", location="lower right")
    axs[0, 1].add_artist(scalebar)

    im10 = axs[1, 0].imshow(I_center, cmap="gray")
    axs[1, 0].set_title(r"$\theta = 0^\circ$")
    fig.colorbar(im10, ax=axs[1, 0])
    scalebar = ScaleBar(config["dx"], "m", location="lower right")
    axs[1, 0].add_artist(scalebar)

    im11 = axs[1, 1].imshow(I_shift, cmap="gray")
    axs[1, 1].set_title(rf"$\theta = {theta_max:.2e}^\circ$")
    fig.colorbar(im11, ax=axs[1, 1])
    scalebar = ScaleBar(config["dx"], "m", location="lower right")
    axs[1, 1].add_artist(scalebar)

    plt.show()
