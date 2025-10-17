from torch.fft import fft2, ifft2, fftshift, ifftshift
import torch


def F(x):
    return fftshift(fft2(x))


def iF(x):
    return ifft2(ifftshift(x))


def forward_model(sample: torch.Tensor, theta: torch.Tensor, config: dict):
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
            torch.sin(theta_rad[0]) * config["X_m"]
            + torch.sin(theta_rad[1]) * config["Y_m"]
        )
    )  # tilted plane wave

    # === Simulate FPM measurement ===
    field = sample * E_illum
    spectrum = F(field)
    filtered = spectrum * config["M"]
    image = torch.abs(iF(filtered)) ** 2
    return image
