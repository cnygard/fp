"""Differentiable Torch optical operators, independent of reconstruction algorithms."""

from math import copysign, floor, isclose, pi, prod

import torch

from .config import OpticalConfig, ReconstructionGrid
from .data import MeasurementSet, wavevectors_tensor
from .fft import fft2c, ifft2c
from .tensors import TorchConfig


def _round_away_from_zero(value):
    return int(copysign(floor(abs(value) + 0.5), value))


class FourierPtychographyModel:
    """Centered object spectrum -> cropped pupil-filtered measured fields.

    Image axes are (y,x), geometry columns are (kx,ky) in rad/m. A measured
    field is ifft2c(pupil * spectrum_patch) / upsample**2. Geometry is rounded
    once on CPU; numerical operations stay on the configured Torch device.
    Optional pupil arguments let future solvers own and differentiate a pupil
    estimate without mutating the fixed model support or another solver's state.
    """

    def __init__(
        self,
        optics: OpticalConfig,
        grid: ReconstructionGrid,
        wavevectors_rad_per_m,
        *,
        compute: TorchConfig = TorchConfig(),
    ):
        self.optics, self.grid = optics, grid
        self.requested_compute = compute
        self.compute = compute.resolve()
        self.device = torch.device(self.compute.device)
        self.real_dtype = self.compute.real_dtype
        self.complex_dtype = self.compute.complex_dtype
        self.wavevectors_rad_per_m = wavevectors_tensor(wavevectors_rad_per_m)
        height, width = grid.measurement_shape
        out_h, out_w = grid.object_shape
        # Precompute support in CPU double precision to keep boundary decisions
        # consistent across devices, then transfer once for all subsequent FFTs.
        ky = (
            2
            * pi
            * torch.fft.fftshift(
                torch.fft.fftfreq(height, optics.pixel_size_m, dtype=torch.float64)
            )
        )
        kx = (
            2
            * pi
            * torch.fft.fftshift(
                torch.fft.fftfreq(width, optics.pixel_size_m, dtype=torch.float64)
            )
        )
        cutoff = optics.numerical_aperture * optics.wavenumber_rad_per_m
        self.pupil_support = (torch.hypot(ky[:, None], kx[None, :]) <= cutoff).to(
            self.device
        )
        self.pupil = self.pupil_support.to(self.complex_dtype)
        dkx = 2 * pi / (optics.pixel_size_m * width)
        dky = 2 * pi / (optics.pixel_size_m * height)
        patches = []
        for index, (shift_x, shift_y) in enumerate(self.wavevectors_rad_per_m.tolist()):
            left = out_w // 2 + _round_away_from_zero(-shift_x / dkx) - width // 2
            top = out_h // 2 + _round_away_from_zero(-shift_y / dky) - height // 2
            if left < 0 or top < 0 or left + width > out_w or top + height > out_h:
                raise ValueError(
                    f"frame {index}: frequency patch exceeds the object grid; increase upsample"
                )
            patches.append((slice(top, top + height), slice(left, left + width)))
        self.patches = tuple(patches)

    def as_complex(self, values):
        return torch.as_tensor(values, device=self.device, dtype=self.complex_dtype)

    def validate_measurements(self, measurements: MeasurementSet):
        if measurements.intensities.shape != (
            len(self.patches),
            *self.grid.measurement_shape,
        ):
            raise ValueError("measurement shape does not match the optical model")
        if not torch.equal(
            measurements.wavevectors_rad_per_m, self.wavevectors_rad_per_m
        ):
            raise ValueError(
                "measurement geometry/order does not match the optical model"
            )
        if not isclose(
            measurements.pixel_size_m,
            self.optics.pixel_size_m,
            rel_tol=1e-12,
            abs_tol=0,
        ):
            raise ValueError(
                "measurement pixel spacing does not match the optical model"
            )

    def _pupil(self, pupil):
        pupil = self.pupil if pupil is None else self.as_complex(pupil)
        if pupil.shape != self.grid.measurement_shape:
            raise ValueError("pupil must match the measurement shape")
        return pupil

    def field_from_spectrum(self, object_spectrum, frame_index: int, *, pupil=None):
        spectrum = self.as_complex(object_spectrum)
        if spectrum.shape != self.grid.object_shape:
            raise ValueError(
                "object spectrum shape does not match the reconstruction grid"
            )
        return (
            ifft2c(spectrum[self.patches[frame_index]] * self._pupil(pupil))
            / self.grid.field_scale
        )

    def forward_fields(self, object_spectrum, *, pupil=None):
        """Linear field operator A on the centered spectrum; preserves autograd."""
        return torch.stack(
            [
                self.field_from_spectrum(object_spectrum, i, pupil=pupil)
                for i in range(len(self.patches))
            ]
        )

    def adjoint_fields(self, fields, *, pupil=None):
        """Hermitian adjoint A*, including FFT normalization and overlap sums.

        This is an adjoint, not an inverse, and does not average over frames.
        It supports complex inner-product and gradient checks for Wirtinger flow.
        """
        fields = self.as_complex(fields)
        if fields.shape != (len(self.patches), *self.grid.measurement_shape):
            raise ValueError("fields must match all measurement frames")
        filtered = fft2c(fields) * self._pupil(pupil).conj()
        filtered = filtered / (
            prod(self.grid.measurement_shape) * self.grid.field_scale
        )
        spectrum = torch.zeros(
            self.grid.object_shape, device=self.device, dtype=self.complex_dtype
        )
        for index, patch in enumerate(self.patches):
            spectrum[patch] += filtered[index]
        return spectrum

    def predict_intensities(self, object_field, *, pupil=None):
        object_field = self.as_complex(object_field)
        if (
            object_field.shape != self.grid.object_shape
            or not torch.isfinite(object_field).all()
        ):
            raise ValueError(
                "object field must be finite and match the reconstruction grid"
            )
        return self.forward_fields(fft2c(object_field), pupil=pupil).abs().square()

    def intensity_loss(self, object_spectrum, intensities, *, pupil=None):
        """Differentiable half mean-squared intensity error for future optimizers."""
        prediction = self.forward_fields(object_spectrum, pupil=pupil).abs().square()
        observed = torch.as_tensor(
            intensities, device=self.device, dtype=self.real_dtype
        )
        if observed.shape != prediction.shape:
            raise ValueError("intensities must match all measurement frames")
        return 0.5 * (prediction - observed).square().mean()

    @torch.no_grad()
    def amplitude_rmse(self, object_spectrum, intensities, *, pupil=None):
        observed = torch.as_tensor(
            intensities, device=self.device, dtype=self.real_dtype
        )
        if observed.shape != (len(self.patches), *self.grid.measurement_shape):
            raise ValueError("intensities must match all measurement frames")
        error = torch.zeros((), device=self.device, dtype=self.real_dtype)
        for index in range(len(self.patches)):
            residual = (
                self.field_from_spectrum(object_spectrum, index, pupil=pupil).abs()
                - observed[index].sqrt()
            )
            error += residual.square().sum()
        return (error / observed.numel()).sqrt().item()
