"""Independent gradients, descent, scale, precision, and solver isolation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from fp import SolverOptions, TorchConfig, FourierPtychographyModel, OpticalConfig, ReconstructionGrid, WirtingerFlowReconstructor
from fp.fft import fft2c
from fp.reconstructors.wirtinger import intensity_loss_gradient
from fp.results import save_result
from fp.simulation import simulate_measurements


def fixture(precision="float64"):
    optics = OpticalConfig(1.85e-6, 532e-9, .6, 2.)
    dk = 2*np.pi/(8*optics.pixel_size_m)
    model = FourierPtychographyModel(
        optics, ReconstructionGrid((8, 8), 2), [[0, 0], [dk, 0], [0, dk], [-dk, -dk]],
        compute=TorchConfig("cpu", precision),
    )
    y, x = torch.meshgrid(torch.linspace(-1, 1, 16), torch.linspace(-1, 1, 16), indexing="ij")
    truth = (.3 + .5*torch.exp(-5*(x*x+y*y))) * torch.exp(.2j*x)
    return model, simulate_measurements(truth, model)


class WirtingerTests(unittest.TestCase):
    def test_streaming_loss_and_gradient_match_autograd_and_adjoint(self):
        model, data = fixture()
        generator = torch.Generator().manual_seed(32)
        spectrum = torch.randn(model.grid.object_shape, dtype=torch.complex128, generator=generator, requires_grad=True)
        pupil = torch.randn(model.grid.measurement_shape, dtype=torch.complex128, generator=generator)
        expected_loss = model.intensity_loss(spectrum, data.intensities, pupil=pupil)
        expected, = torch.autograd.grad(expected_loss, spectrum)
        with torch.no_grad():
            loss, gradient = intensity_loss_gradient(model, spectrum, data.intensities, pupil)
            fields = model.forward_fields(spectrum, pupil=pupil)
            adjoint = 2 * model.adjoint_fields((fields.abs().square()-data.intensities)*fields, pupil=pupil) / data.intensities.numel()
        torch.testing.assert_close(loss, expected_loss, atol=1e-15, rtol=1e-13)
        torch.testing.assert_close(gradient, expected, atol=1e-15, rtol=1e-12)
        torch.testing.assert_close(gradient, adjoint, atol=1e-15, rtol=1e-12)

    def test_one_step_matches_independent_complex_gradient_update(self):
        model, data = fixture()
        initial = torch.full(model.grid.object_shape, .4, dtype=model.complex_dtype)
        spectrum = fft2c(initial).requires_grad_()
        loss = model.intensity_loss(spectrum, data.intensities)
        gradient, = torch.autograd.grad(loss, spectrum)
        states = []
        result = WirtingerFlowReconstructor(model, SolverOptions(1)).reconstruct(
            data, initial_field=initial, observer=states.append)
        step = states[-1].extras["step_size"]
        torch.testing.assert_close(fft2c(result.object_field), spectrum - step*gradient, atol=1e-12, rtol=1e-12)
        self.assertLess(model.intensity_loss(fft2c(result.object_field), data.intensities).item(), loss.item())

    def test_converges_and_all_accepted_steps_reduce_intensity_loss(self):
        model, data = fixture()
        states, callbacks = [], []
        solver = WirtingerFlowReconstructor(model, SolverOptions(80))
        result = solver.reconstruct(data, observer=states.append, callback=callbacks.append, observer_interval=7)
        losses = [model.intensity_loss(state.spectrum, data.intensities, pupil=state.pupil).item() for state in states]
        self.assertTrue(all(b <= a for a,b in zip(losses, losses[1:])))
        self.assertLess(losses[-1], losses[0]*.05)
        self.assertLess(result.history[-1].amplitude_rmse, result.history[0].amplitude_rmse*.25)
        self.assertEqual([entry.epoch for entry in callbacks], list(range(1, 81)))
        self.assertFalse(result.settings["pupil_recovery"]["enabled"])
        self.assertFalse(result.object_field.requires_grad)

    def test_backtracking_with_large_learning_rate_and_hdr_scale(self):
        model, data = fixture()
        from fp import MeasurementSet
        for scale in (.0001, 10000.):
            scaled = MeasurementSet(data.intensities*scale, data.wavevectors_rad_per_m, data.pixel_size_m)
            initial = torch.full(model.grid.object_shape, .4*np.sqrt(scale), dtype=model.complex_dtype)
            result = WirtingerFlowReconstructor(model, SolverOptions(3, 1e5)).reconstruct(scaled, initial_field=initial)
            initial_loss = model.intensity_loss(fft2c(initial), scaled.intensities)
            final_loss = model.intensity_loss(fft2c(result.object_field), scaled.intensities)
            self.assertTrue(torch.isfinite(result.object_field).all())
            self.assertLess(final_loss, initial_loss)

    def test_fixed_pupil_inputs_diagnostics_and_export_are_owned(self):
        model, data = fixture()
        pupil = model.pupil.clone() * torch.exp(torch.tensor(.1j))
        original = pupil.clone()
        solver = WirtingerFlowReconstructor(model, SolverOptions(3))
        expected = solver.reconstruct(data, initial_pupil=pupil)
        def corrupt_snapshot(state):
            state.spectrum.zero_()
            state.pupil.zero_()
            for value in state.extras.values():
                value.zero_()
        result = solver.reconstruct(data, initial_pupil=pupil, observer=corrupt_snapshot)
        torch.testing.assert_close(result.object_field, expected.object_field, atol=0, rtol=0)
        torch.testing.assert_close(result.pupil, original)
        torch.testing.assert_close(pupil, original)
        torch.testing.assert_close(model.pupil, model.pupil_support.to(model.complex_dtype))
        with TemporaryDirectory() as temporary:
            with np.load(save_result(result, Path(temporary)/"result.npz"), allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["pupil"], original.numpy())
        for invalid in (torch.ones((2, 2)), torch.full(model.grid.measurement_shape, torch.nan)):
            with self.assertRaisesRegex(ValueError, "initial_pupil"):
                solver.reconstruct(data, initial_pupil=invalid)
        with self.assertRaises(ValueError):
            solver.reconstruct(data, observer_interval=0)

    def test_cli_headless_debug_renders_batch_updates_and_exports(self):
        from contextlib import redirect_stdout
        import io
        from fp.cli import parse_args, run
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "0000.npy", np.full((8, 8), .25))
            args = parse_args([
                temporary, "-w", "79.99", "-t", "28", "-n", "1", "-r", "8", "-e", "2",
                "--algorithm", "wirtinger_flow", "--device", "cpu", "--precision", "float32",
                "--debug-dir", str(directory / "debug"), "--debug-views", "spectrum", "residual", "update",
                "--output", str(directory / "result.npz"),
            ])
            with redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(result.object_field.dtype, torch.complex64)
            self.assertEqual(len(list((directory / "debug").glob("*.png"))), 3)
            self.assertTrue((directory / "result.npz").is_file())

    def test_zero_data_and_float32_are_finite(self):
        from fp import MeasurementSet
        for precision, expected_dtype in (("float64", torch.complex128), ("float32", torch.complex64)):
            model, data = fixture(precision)
            zero = MeasurementSet(torch.zeros_like(data.intensities), data.wavevectors_rad_per_m, data.pixel_size_m)
            for measured in (data, zero):
                result = WirtingerFlowReconstructor(model, SolverOptions(5)).reconstruct(measured)
                self.assertEqual(result.object_field.dtype, expected_dtype)
                self.assertTrue(torch.isfinite(result.object_field).all())
                if measured is zero:
                    self.assertEqual(result.history[-1].amplitude_rmse, 0.)


if __name__ == "__main__":
    unittest.main()
