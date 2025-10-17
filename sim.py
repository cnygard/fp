import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
from torch.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import torch


km = 1e3
m = 1
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9


def F(x):
    return fftshift(fft2(x))


def iF(x):
    return ifft2(ifftshift(x))


def propagate(sample: torch.Tensor, theta: torch.Tensor, config: dict):
    """Applys Fourier ptychography forward imaging model.

    Args:
        sample: Complex sample to propagate
        theta: 2D rotation (i.e., [theta_y, theta_x]) in degrees
        config: Optical system configuration info

    Returns:
        Intensity measurement at the image plane.
    """
    theta_rad = torch.deg2rad(theta)

    # === Compute lateral wavevector shift ===
    E_illum = torch.exp(
        1j
        * config["k"]
        * (
            torch.sin(theta_rad[0]) * config["Y_m"]
            + torch.sin(theta_rad[1]) * config["X_m"]
        )
    )  # tilted plane wave

    # === Simulate FPM measurement ===
    field = sample * E_illum
    spectrum = F(field)
    filtered = spectrum * M
    image = torch.abs(iF(filtered)) ** 2
    return image


# Microscopy configuration
FPM_CONFIG = {
    "title": "Microscopy",
    "dx": 0.5 * um,
    "aperture_diameter": 8 * mm,
    "object_distance": 4 * cm,
    "NA_illum": 0.3,
}

# Long-range imaging configuration
FPI_CONFIG = {
    "title": "Long-range imaging",
    "dx": 10 * cm,
    "aperture_diameter": 50 * cm,
    "object_distance": 515 * km,  # Hubble Space Telescope distance
    "NA_illum": 1.5e-6,
}


if __name__ == "__main__":
    # Set experimental configuration
    config = FPM_CONFIG
    print(config)

    # === Load and preprocess image ===
    sample = cv2.imread("/Users/m/Downloads/usaf.jpg", 0).astype(np.float32)
    sample = sample / 255.0  # normalize
    sample = torch.from_numpy(sample)

    config["N"] = sample.shape[0]  # assume square image
    print("Object width", config["dx"] * config["N"])
    config["wavelength"] = 532 * nm  # wavelength
    config["k"] = 2 * np.pi / config["wavelength"]  # wave number

    # === Define pupil mask (circular aperture) ===
    NA_obj = config["aperture_diameter"] / (2 * config["object_distance"])
    print(f"Objective NA {NA_obj:.2e}")
    pupil_radius = int(NA_obj * config["N"] * config["dx"] / config["wavelength"])
    print(f"Aperture radius (pixels): {pupil_radius}")

    x = torch.arange(-config["N"] // 2, config["N"] // 2)
    Y, X = torch.meshgrid(x, x, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)
    M = R < pupil_radius  # circular aperture mask

    # === Define illumination angle ===
    theta_max = np.rad2deg(np.arcsin(config["NA_illum"]))
    print(f"Max rotation (deg) {theta_max:.2e}")
    print(
        f"Max scan distance (m) {config['object_distance'] * np.tan(np.arcsin(config['NA_illum'])):.2e}"
    )
    config["X_m"] = X * config["dx"]
    config["Y_m"] = Y * config["dx"]

    I_center = propagate(sample, torch.tensor([0, 0]), config)
    I_shift = propagate(sample, torch.tensor([0, theta_max / 4]), config)

    # === Display ===
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Config: {config['title']}")

    im00 = axs[0, 0].imshow(sample, cmap="gray")
    axs[0, 0].set_title(f"Sample (z={config['object_distance']:.2e}m)")
    fig.colorbar(im00, ax=axs[0, 0])
    scalebar = ScaleBar(config["dx"], "m", location="lower right")
    axs[0, 0].add_artist(scalebar)

    im01 = axs[0, 1].imshow(M)
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
