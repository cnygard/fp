"""Centered Torch FFTs, default backward normalization, spatial axes only."""

import torch


def fft2c(field) -> torch.Tensor:
    field = torch.as_tensor(field)
    return torch.fft.fftshift(torch.fft.fft2(field, dim=(-2, -1)), dim=(-2, -1))


def ifft2c(spectrum) -> torch.Tensor:
    spectrum = torch.as_tensor(spectrum)
    return torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1))
