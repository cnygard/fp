import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2

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


def propagate(theta_deg):
    theta_rad = np.deg2rad(theta_deg)

    # === Compute lateral wavevector shift ===
    kx = k * np.sin(theta_rad)  # in rad/m
    E_illum = np.exp(1j * kx * X_m)  # tilted plane wave

    # === Simulate FPM measurement ===
    field = img * E_illum
    spectrum = F(field)
    filtered = spectrum * M
    recon = np.abs(iF(filtered)) ** 2
    return recon


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
    config = FPI_CONFIG
    print(config)

    # === Load and preprocess image ===
    img = cv2.imread("/Users/m/Downloads/usaf.jpg", 0).astype(np.float32)
    img = img / 255.0  # normalize

    N = img.shape[0]  # assume square image
    dx = config["dx"]  # sampling step in object space
    print("Object width", dx * N)
    wavelength = 532 * nm  # wavelength
    k = 2 * np.pi / wavelength  # wave number

    # === Define pupil mask (circular aperture) ===
    aperture_diameter = config["aperture_diameter"]
    object_distance = config["object_distance"]
    NA_obj = aperture_diameter / (2 * object_distance)
    print(f"Objective NA {NA_obj:.2e}")
    pupil_radius = int(NA_obj * N * dx / wavelength)  # in pixels
    print(f"Aperture radius (pixels): {pupil_radius}")

    x = np.arange(-N // 2, N // 2)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    M = R < pupil_radius  # circular aperture mask

    # === Define illumination angle ===
    NA_illum = config["NA_illum"]  # illumination numerical aperture
    theta_max = np.rad2deg(np.arcsin(NA_illum))
    print(f"Max rotation (deg) {theta_max:.2e}")
    print(f"Max scan distance (m) {object_distance * np.tan(np.arcsin(NA_illum)):.2e}")
    X_m = X * dx

    I_center = propagate(0)
    I_shift = propagate(theta_max / 4)

    # === Display ===
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Config: {config['title']}")

    im00 = axs[0, 0].imshow(img, cmap="gray")
    axs[0, 0].set_title(f"Sample (z={object_distance:.2e}m)")
    fig.colorbar(im00, ax=axs[0, 0])
    scalebar = ScaleBar(dx, "m", location="lower right")
    axs[0, 0].add_artist(scalebar)

    im01 = axs[0, 1].imshow(M)
    axs[0, 1].set_title("Pupil mask")
    fig.colorbar(im01, ax=axs[0, 1])
    scalebar = ScaleBar(dx, "m", location="lower right")
    axs[0, 1].add_artist(scalebar)

    im10 = axs[1, 0].imshow(I_center, cmap="gray")
    axs[1, 0].set_title(r"$\theta = 0^\circ$")
    fig.colorbar(im10, ax=axs[1, 0])
    scalebar = ScaleBar(dx, "m", location="lower right")
    axs[1, 0].add_artist(scalebar)

    im11 = axs[1, 1].imshow(I_shift, cmap="gray")
    axs[1, 1].set_title(rf"$\theta = {theta_max:.2e}^\circ$")
    fig.colorbar(im11, ax=axs[1, 1])
    scalebar = ScaleBar(dx, "m", location="lower right")
    axs[1, 1].add_artist(scalebar)

    plt.show()
