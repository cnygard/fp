"""Torch conversion and explicit compute-device/precision policy."""

from dataclasses import dataclass

import torch


def real_tensor(values, *, device=None, dtype=None) -> torch.Tensor:
    tensor = torch.as_tensor(values, device=device)
    if tensor.is_complex():
        raise ValueError("expected real values")
    if dtype is None:
        dtype = tensor.dtype if tensor.dtype in (torch.float32, torch.float64) else torch.float64
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("real dtype must be torch.float32 or torch.float64")
    return torch.as_tensor(values, device=device, dtype=dtype)


@dataclass(frozen=True)
class TorchConfig:
    """Prefer accelerators, then the most accurate supported precision on each.

    auto device order is CUDA, MPS, CPU. auto precision order is float64 /
    complex128, then float32 / complex64. Explicit selections are strict:
    requesting float64 never silently becomes float32, and an explicit device
    never silently becomes CPU. Resolution probes the installed Torch runtime.
    """

    device: str = "auto"
    precision: str = "auto"

    def __post_init__(self) -> None:
        if self.precision not in ("auto", "float32", "float64"):
            raise ValueError("precision must be auto, float32, or float64")
        if self.device != "auto" and torch.device(self.device).type not in (
            "cpu",
            "cuda",
            "mps",
        ):
            raise ValueError("device must be cpu, cuda, mps, or auto")

    @property
    def real_dtype(self) -> torch.dtype:
        if self.precision == "auto":
            return self.resolve().real_dtype
        return {"float32": torch.float32, "float64": torch.float64}[self.precision]

    @property
    def complex_dtype(self) -> torch.dtype:
        if self.precision == "auto":
            return self.resolve().complex_dtype
        return {"float32": torch.complex64, "float64": torch.complex128}[self.precision]

    @staticmethod
    def _probe(device, precision):
        dtype = {"float32": torch.complex64, "float64": torch.complex128}[precision]
        probe = torch.ones((4, 4), device=device, dtype=dtype)
        reconstructed = torch.fft.ifft2(torch.fft.fft2(probe))
        if not torch.isfinite(reconstructed).all().item():
            raise RuntimeError("complex FFT round trip produced nonfinite values")

    def resolve(self) -> TorchConfig:
        """Return an explicit device/precision pair, without changing global defaults."""
        devices = [self.device]
        if self.device == "auto":
            devices = []
            if torch.cuda.is_available():
                devices.append("cuda")
            if torch.backends.mps.is_available():
                devices.append("mps")
            devices.append("cpu")
        precisions = ("float64", "float32") if self.precision == "auto" else (self.precision,)
        failures = []
        for candidate in devices:
            for precision in precisions:
                try:
                    device = torch.device(candidate)
                    self._probe(device, precision)
                    return TorchConfig(device=str(device), precision=precision)
                except (
                    RuntimeError,
                    NotImplementedError,
                    AssertionError,
                    TypeError,
                ) as error:
                    failures.append(f"{candidate}/{precision}: {error}")
        raise ValueError(
            "no requested device/precision supports complex FFTs; " + "; ".join(failures)
        )

    def resolve_device(self) -> torch.device:
        """Compatibility helper; model construction resolves the complete pair once."""
        return torch.device(self.resolve().device)
