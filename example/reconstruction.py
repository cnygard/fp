import torch
import numpy as np
from forward_model import *
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from config import *


def iterative_reconstruction(gt_sample, Theta, num_epochs):
    # Init guesses
    dc = forward_model(gt_sample, torch.tensor([0, 0]), config)
    sample = torch.nn.Parameter(dc)
    optimizer = torch.optim.Adam([sample], lr=1e-2)

    prog_bar = tqdm(range(num_epochs))
    for _ in prog_bar:
        for theta in Theta:
            image = forward_model(sample, theta, config)
            target = forward_model(gt_sample, theta, config)
            loss = torch.nn.functional.mse_loss(image, target)
            loss.backward()
            optimizer.step()
            error = torch.nn.functional.mse_loss(sample, gt_sample)
            prog_bar.set_description(f"Error: {error.item():.4f}")
            optimizer.zero_grad()
    return sample


if __name__ == "__main__":
    # === Load and preprocess image ===
    sample = cv2.imread("/Users/m/Downloads/usaf.jpg", 0).astype(np.float32)
    sample = cv2.resize(sample, (256, 256))
    sample = sample / 255.0  # normalize
    sample = torch.from_numpy(sample)

    config = setup_config(sample, FPM_CONFIG)

    theta_max = np.rad2deg(np.arcsin(config["NA_illum"]))
    print(f"Max rotation (deg) {theta_max:.2e}")
    theta = torch.linspace(-theta_max, theta_max, 10)
    theta_y, theta_x = torch.meshgrid(theta, theta, indexing="ij")
    Theta = torch.stack([theta_y, theta_x], dim=-1).reshape(-1, 2)

    pred = iterative_reconstruction(sample, Theta, num_epochs=25)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(forward_model(sample, torch.tensor([0, 0]), config))
    axs[0].set_title("Measurement")
    axs[1].imshow(pred.detach().cpu())
    axs[1].set_title("FP Reconstruction")
    axs[2].imshow(sample)
    axs[2].set_title("GT")
    plt.show()
