"""Hardware-independent compute policy checks and effective run reporting."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fp import FourierPtychographyModel, OpticalConfig, ReconstructionGrid, TorchConfig
from fp.cli import parse_args, run
from fp.fft import fft2c


class ComputePolicyTests(unittest.TestCase):
    def resolve(self, request, supported, *, cuda=True, mps=True):
        attempts = []
        def probe(device, precision):
            pair = (str(device), precision)
            attempts.append(pair)
            if pair not in supported:
                raise RuntimeError("unsupported test device/precision")
        with patch("torch.cuda.is_available", return_value=cuda), \
             patch("torch.backends.mps.is_available", return_value=mps), \
             patch.object(TorchConfig, "_probe", side_effect=probe):
            result = request.resolve()
        return result, attempts

    def test_default_prefers_cuda_double(self):
        result, attempts = self.resolve(TorchConfig(), {("cuda", "float64")})
        self.assertEqual(result, TorchConfig("cuda", "float64"))
        self.assertEqual(attempts, [("cuda", "float64")])

    def test_device_priority_precedes_precision_on_slower_devices(self):
        result, attempts = self.resolve(
            TorchConfig(), {("cuda", "float32"), ("mps", "float64"), ("cpu", "float64")},
        )
        self.assertEqual(result, TorchConfig("cuda", "float32"))
        self.assertEqual(attempts, [("cuda", "float64"), ("cuda", "float32")])

    def test_mps_single_precision_is_selected_before_cpu_double(self):
        result, attempts = self.resolve(
            TorchConfig(), {("mps", "float32"), ("cpu", "float64")}, cuda=False,
        )
        self.assertEqual(result, TorchConfig("mps", "float32"))
        self.assertEqual(attempts, [("mps", "float64"), ("mps", "float32")])

    def test_cpu_double_is_used_when_no_accelerator_is_available(self):
        result, attempts = self.resolve(
            TorchConfig(), {("cpu", "float64")}, cuda=False, mps=False,
        )
        self.assertEqual(result, TorchConfig("cpu", "float64"))
        self.assertEqual(result.real_dtype, torch.float64)
        self.assertEqual(result.complex_dtype, torch.complex128)
        self.assertEqual(attempts, [("cpu", "float64")])

    def test_unsupported_accelerator_falls_back_to_next_device(self):
        result, attempts = self.resolve(
            TorchConfig(), {("cpu", "float64")}, cuda=False,
        )
        self.assertEqual(result.device, "cpu")
        self.assertEqual(attempts[-1], ("cpu", "float64"))

    def test_explicit_precision_is_never_downgraded(self):
        result, attempts = self.resolve(
            TorchConfig(precision="float64"), {("mps", "float32"), ("cpu", "float64")}, cuda=False,
        )
        self.assertEqual(result, TorchConfig("cpu", "float64"))
        self.assertEqual(attempts, [("mps", "float64"), ("cpu", "float64")])

    def test_explicit_device_can_choose_its_own_highest_precision(self):
        result, attempts = self.resolve(TorchConfig(device="mps"), {("mps", "float32")})
        self.assertEqual(result, TorchConfig("mps", "float32"))
        self.assertEqual(attempts, [("mps", "float64"), ("mps", "float32")])

    def test_explicit_device_never_silently_falls_back(self):
        with self.assertRaisesRegex(ValueError, "mps/float64"):
            self.resolve(TorchConfig(device="mps", precision="float64"), {("cpu", "float64")})

    def test_explicit_indexed_cuda_and_single_precision_are_preserved(self):
        result, attempts = self.resolve(TorchConfig("cuda:2", "float32"), {("cuda:2", "float32")})
        self.assertEqual(result, TorchConfig("cuda:2", "float32"))
        self.assertEqual(attempts, [("cuda:2", "float32")])


class ReportingAndMetricTests(unittest.TestCase):
    def test_cli_defaults_and_overrides(self):
        with TemporaryDirectory() as temporary:
            arguments = [temporary, "-w", "79.99", "-t", "28"]
            args = parse_args(arguments)
            self.assertEqual((args.device, args.precision), ("auto", "auto"))
            args = parse_args(arguments + ["--device", "cpu", "--precision", "float32"])
            self.assertEqual((args.device, args.precision), ("cpu", "float32"))

    def test_startup_report_prints_effective_sampling_and_actual_compute_before_epochs(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "0000.npy", np.ones((16, 16)))
            args = parse_args([
                temporary, "-w", "79.99", "-t", "28", "-n", "1", "-r", "8", "-e", "1",
                "--device", "cpu", "--precision", "float64",
            ])
            output = io.StringIO()
            with redirect_stdout(output):
                result = run(args)
            text = output.getvalue()
            for fragment in (
                "fp reconstruction", "source: array_sim", "1 x 8 x 8", "wavelength 532 nm",
                "aperture diameter 79.99 mm", "object distance 2 m", "NA", "measurement 3.7 um",
                "reconstructed object 1.85 um", "Field of view", "8 x 8 -> 16 x 16",
                "Solver: savi", "object learning rate 1", "Pupil: joint recovery enabled",
                "Compute: cpu", "real float64, complex complex128", "Torch:", "Debug: off",
                "Result file: disabled",
            ):
                self.assertIn(fragment, text)
            self.assertLess(text.index("fp reconstruction"), text.index("Epoch 1/1"))
            self.assertEqual(result.settings["compute"], {"device": "cpu", "precision": "float64"})

    def test_metric_uses_supplied_pupil_estimate(self):
        model = FourierPtychographyModel(
            OpticalConfig(1.85e-6, 532e-9, .6, 2.), ReconstructionGrid((6, 8)), [[0, 0]],
            compute=TorchConfig("cpu", "float64"),
        )
        field = torch.ones(model.grid.object_shape, dtype=model.complex_dtype)
        spectrum = fft2c(field)
        estimate = 1.5 * model.pupil
        measured = model.predict_intensities(field, pupil=estimate)
        self.assertAlmostEqual(model.amplitude_rmse(spectrum, measured, pupil=estimate), 0, places=12)
        self.assertGreater(model.amplitude_rmse(spectrum, measured), .4)
        with self.assertRaisesRegex(ValueError, "intensities must match"):
            model.amplitude_rmse(spectrum, measured[0], pupil=estimate)


if __name__ == "__main__":
    unittest.main()
