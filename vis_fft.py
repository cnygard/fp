"""Compare and visualize the magnitude spectra of two grayscale images."""

import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE1_PATH = "../bench_results/usaf_11_wide_slit/0001.jpg"
IMAGE2_PATH = "../bench_results/usaf_11_wide_slit/0006.jpg"
SAVE_PATH = None


def compute_log_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """Return log magnitude of the centered FFT of sqrt(image)."""
    transformed = np.fft.fftshift(np.fft.fft2(np.sqrt(image.astype(np.float64))))
    return np.abs(np.log(transformed))


def load_grayscale(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def main() -> int:
    img1 = load_grayscale(IMAGE1_PATH)
    img2 = load_grayscale(IMAGE2_PATH)

    spec1 = compute_log_magnitude_spectrum(img1)
    spec2 = compute_log_magnitude_spectrum(img2)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(img1, cmap="gray")
    axes[0, 1].imshow(img2, cmap="gray")
    axes[1, 0].imshow(spec1, cmap="gray")
    axes[1, 1].imshow(spec2, cmap="gray")

    fig.tight_layout()

    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {SAVE_PATH}")

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
