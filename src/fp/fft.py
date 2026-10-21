"""Centered spectra with NumPy's default FFT normalization, spatial axes only."""

import numpy as np


def fft2c(field):
    return np.fft.fftshift(np.fft.fft2(field, axes=(-2, -1)), axes=(-2, -1))


def ifft2c(spectrum):
    return np.fft.ifft2(np.fft.ifftshift(spectrum, axes=(-2, -1)), axes=(-2, -1))
