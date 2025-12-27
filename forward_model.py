from torch.fft import fft2, ifft2, fftshift, ifftshift
import torch
import matplotlib.pyplot as plt


def F(x):
    return fftshift(fft2(x))


def iF(x):
    return ifft2(ifftshift(x))


def forward_model(sample: torch.Tensor, theta: torch.Tensor, config: dict, targ: torch.Tensor):
    """Applies Fourier ptychography forward imaging model.

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
    # import pdb; pdb.set_trace()

    # === Simulate FPM measurement ===
    field = sample * E_illum
    spectrum = F(field)
    filtered = spectrum * config["M"]
    image = torch.abs(iF(filtered)) ** 2
    filtered = filtered.detach().numpy().astype(float) > 0.01
    spectrum = spectrum.detach().numpy().astype(float) > 0.01

    # fig, axs = plt.subplots(3, 2)
    # axs[0,0].imshow(field.detach().numpy().astype(float), cmap='gray')
    # axs[0,0].set_title("field")
    # # axs[0,1].imshow(spectrum.detach().numpy().astype(float), cmap='gray')
    # axs[0,1].imshow(spectrum, cmap='gray')
    # axs[0,1].set_title("spectrum")
    # # axs[1,0].imshow(filtered.detach().numpy().astype(float), cmap='gray')
    # axs[1,0].imshow(filtered, cmap='gray')
    # axs[1,0].set_title("filtered")
    # axs[1,1].imshow(image.detach().numpy().astype(float), cmap='gray')
    # axs[1,1].set_title("image")
    #
    # axs[2,0].imshow(targ.detach().numpy().astype(float), cmap='gray')
    # axs[2,0].set_title("target")
    # targF = F(targ).detach().numpy().astype(float) > 0.01
    # axs[2,1].imshow(targF, cmap='gray')
    # axs[2,1].set_title("target")

    plt.show()

    return image
