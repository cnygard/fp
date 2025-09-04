import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2


def F(x):
    return fftshift(fft2(x))


def iF(x):
    return ifft2(ifftshift(x))


if __name__ == "__main__":
    # === Load and preprocess image ===
    img = cv2.imread("/Users/m/Downloads/usaf.jpg", 0).astype(np.float32)
    img = img / 255.0  # normalize

    N = img.shape[0]  # assume square image
    dx = 0.5e-6  # sampling step in object space [meters]
    print("Object width", dx * N)
    wavelength = 532e-9  # wavelength [meters]
    k = 2 * np.pi / wavelength  # wave number

    # === Define pupil mask (circular aperture) ===
    NA_obj = 0.1
    pupil_radius = int(NA_obj * N * dx / wavelength)  # in pixels
    print(f"Aperture radius (pixels): {pupil_radius}")

    x = np.arange(-N // 2, N // 2)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    M = R < pupil_radius  # circular aperture mask

    # === Define illumination angle ===
    NA_illum = 0.3  # illumination numerical aperture
    theta_deg = 17.5  # degrees (arcsin(0.3) ~ 17.5)
    theta_rad = np.deg2rad(theta_deg)

    # === Compute lateral wavevector shift ===
    kx = k * np.sin(theta_rad)  # in rad/m
    # Convert spatial coordinate grid X to meters
    X_m = X * dx
    E_illum = np.exp(1j * kx * X_m)  # tilted plane wave

    # === Simulate FPM measurement ===
    field = img * E_illum
    spectrum = F(field)
    filtered = spectrum * M
    recon = np.abs(iF(filtered)) ** 2

    # === Display ===
    plt.figure(figsize=(6, 6))
    plt.imshow(recon, cmap="gray")
    plt.title(f"FPM Measurement with θ = {theta_deg:.1f}°")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
