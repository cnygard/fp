"""Torch/autograd contracts, Wirtinger operators, and diagnostic isolation."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import torch

from fp import (
    FourierPtychographyModel, OpticalConfig, ReconstructionGrid,
    SaviReconstructor, SolverOptions, TorchConfig,
)
from fp.cli import parse_args, run
from fp.diagnostics import emit_state
from fp.fft import fft2c
from fp.preprocessing import resize_image
from fp.simulation import simulate_measurements
from fp.tensors import real_tensor


def model_fixture(shape=(6, 8), precision="float64", device="cpu"):
    optics = OpticalConfig(1.85e-6, 532e-9, .6, 2.)
    dkx = 2 * torch.pi / (shape[1] * optics.pixel_size_m)
    dky = 2 * torch.pi / (shape[0] * optics.pixel_size_m)
    return FourierPtychographyModel(
        optics, ReconstructionGrid(shape), [[0, 0], [dkx, 0], [0, dky], [-dkx, -dky]],
        compute=TorchConfig(device=device, precision=precision),
    )


class TorchOperatorTests(unittest.TestCase):
    def test_real_conversion_does_not_round_python_geometry_to_float32(self):
        values = [1.123456789012345, 532e-9]
        torch.testing.assert_close(real_tensor(values, dtype=torch.float64),
                                   torch.tensor(values, dtype=torch.float64), rtol=0, atol=0)

    def test_torch_bilinear_resize_matches_cv2(self):
        image = np.random.default_rng(21).uniform(size=(7, 9))
        resized = resize_image(image, (5, 8)).numpy()
        expected = np.empty((5, 8))
        for row in range(5):
            for col in range(8):
                y, x = (row + .5) * 7 / 5 - .5, (col + .5) * 9 / 8 - .5
                y0, x0 = int(y), int(x)
                dy, dx = y - y0, x - x0
                expected[row, col] = (
                    image[y0, x0] * (1 - dy) * (1 - dx)
                    + image[y0 + 1, x0] * dy * (1 - dx)
                    + image[y0, x0 + 1] * (1 - dy) * dx
                    + image[y0 + 1, x0 + 1] * dy * dx
                )
        np.testing.assert_allclose(resized, expected, atol=1e-14, rtol=1e-14)
        # OpenCV quantizes interpolation coefficients even for float64 images.
        np.testing.assert_allclose(resized, cv2.resize(image, (8, 5)), atol=2e-7, rtol=1e-6)

    def test_complex_field_operator_and_adjoint_satisfy_inner_product_identity(self):
        model = model_fixture()
        generator = torch.Generator().manual_seed(21)
        spectrum = torch.randn(model.grid.object_shape, dtype=torch.complex128, generator=generator)
        fields = torch.randn((4, 6, 8), dtype=torch.complex128, generator=generator)
        pupil = torch.randn((6, 8), dtype=torch.complex128, generator=generator)
        left = torch.vdot(model.forward_fields(spectrum, pupil=pupil).flatten(), fields.flatten())
        right = torch.vdot(spectrum.flatten(), model.adjoint_fields(fields, pupil=pupil).flatten())
        torch.testing.assert_close(left, right, atol=1e-13, rtol=1e-12)

    def test_autograd_matches_explicit_wirtinger_intensity_gradient(self):
        model = model_fixture()
        generator = torch.Generator().manual_seed(22)
        spectrum = torch.randn(model.grid.object_shape, dtype=torch.complex128,
                               generator=generator, requires_grad=True)
        observed = torch.rand((4, 6, 8), dtype=torch.float64, generator=generator)
        gradient, = torch.autograd.grad(model.intensity_loss(spectrum, observed), spectrum)
        fields = model.forward_fields(spectrum.detach())
        expected = 2 * model.adjoint_fields((fields.abs().square() - observed) * fields) / observed.numel()
        torch.testing.assert_close(gradient, expected, atol=1e-14, rtol=1e-12)

    def test_loss_gradcheck_for_object_and_independent_pupil_estimate(self):
        model = model_fixture((3, 4))
        generator = torch.Generator().manual_seed(23)
        spectrum = torch.randn(model.grid.object_shape, dtype=torch.complex128,
                               generator=generator, requires_grad=True)
        pupil = torch.randn(model.grid.measurement_shape, dtype=torch.complex128,
                            generator=generator, requires_grad=True)
        observed = torch.rand((4, 3, 4), dtype=torch.float64, generator=generator)
        self.assertTrue(torch.autograd.gradcheck(
            lambda obj, p: model.intensity_loss(obj, observed, pupil=p),
            (spectrum, pupil), fast_mode=True,
        ))
        self.assertTrue(model.pupil_support.any())
        torch.testing.assert_close(model.pupil, model.pupil_support.to(torch.complex128))

    def test_prediction_retains_object_and_pupil_gradients(self):
        model = model_fixture()
        field = torch.ones(model.grid.object_shape, dtype=torch.complex128, requires_grad=True)
        pupil = model.pupil.clone().requires_grad_()
        prediction = model.predict_intensities(field, pupil=pupil)
        object_grad, pupil_grad = torch.autograd.grad(prediction.sum(), (field, pupil))
        self.assertGreater(object_grad.abs().sum().item(), 0)
        self.assertGreater(pupil_grad.abs().sum().item(), 0)

    def test_float32_stays_on_device_and_results_remain_complex64(self):
        model = model_fixture(precision="float32")
        measurements = simulate_measurements(torch.ones(model.grid.object_shape), model)
        result = SaviReconstructor(model, SolverOptions(2)).reconstruct(measurements)
        self.assertEqual(measurements.intensities.dtype, torch.float32)
        self.assertEqual(result.object_field.dtype, torch.complex64)
        self.assertEqual(result.pupil.dtype, torch.complex64)
        self.assertEqual(result.settings["compute"], {"device": "cpu", "precision": "float32"})
        self.assertTrue(torch.isfinite(result.object_field).all())

    def test_optional_accelerator_smoke(self):
        available = [name for name, enabled in (("cuda", torch.cuda.is_available()),
                                                ("mps", torch.backends.mps.is_available())) if enabled]
        if not available:
            self.skipTest("no accelerator available on this host")
        tested = False
        for device in available:
            try:
                model = model_fixture(precision="float32", device=device)
            except ValueError:
                continue
            measurements = simulate_measurements(torch.ones(model.grid.object_shape), model)
            result = SaviReconstructor(model, SolverOptions(1)).reconstruct(measurements)
            self.assertEqual(result.object_field.device.type, device)
            self.assertEqual(result.object_field.dtype, torch.complex64)
            self.assertTrue(torch.isfinite(result.object_field).all())
            tested = True
        if not tested:
            self.skipTest("available accelerator does not support complex FFTs")


class DiagnosticTests(unittest.TestCase):
    def test_snapshots_are_periodic_owned_graph_free_and_do_not_change_results(self):
        model = model_fixture()
        field = torch.ones(model.grid.object_shape, dtype=torch.complex128)
        data = simulate_measurements(field, model)
        solver = SaviReconstructor(model, SolverOptions(2))
        initial_pupil = model.pupil.clone()
        expected = solver.reconstruct(data)
        states = []
        result = solver.reconstruct(data, observer=states.append, observer_interval=3)
        self.assertEqual([state.step for state in states], [0, 3, 4, 6, 8])
        self.assertEqual([state.metrics.epoch for state in states if state.metrics], [0, 1, 2])
        self.assertEqual(states[-1].frame_index, 3)
        self.assertEqual(set(states[-1].extras), {"measurement", "prediction", "residual", "update", "pupil_update"})
        for state in states:
            self.assertFalse(state.spectrum.requires_grad)
            self.assertFalse(state.pupil.requires_grad)
            self.assertNotEqual(state.spectrum.data_ptr(), result.object_field.data_ptr())
        torch.testing.assert_close(expected.object_field, result.object_field, atol=0, rtol=0)
        def corrupt_only_snapshot(state):
            state.spectrum.zero_()
            state.pupil.zero_()
            for value in state.extras.values():
                value.zero_()
        actual = solver.reconstruct(data, observer=corrupt_only_snapshot)
        torch.testing.assert_close(expected.object_field, actual.object_field, atol=0, rtol=0)
        torch.testing.assert_close(model.pupil, initial_pupil)
        with self.assertRaises(ValueError):
            solver.reconstruct(data, observer_interval=0)

    def test_batch_optimizer_can_emit_frame_free_states_and_custom_extras(self):
        states = []
        spectrum = torch.ones((4, 4), dtype=torch.complex128, requires_grad=True)
        gradient = torch.randn_like(spectrum)
        emit_state(states.append, epoch=1, step=1, spectrum=spectrum,
                   pupil=spectrum[:2, :2], extras={"gradient": gradient})
        self.assertIsNone(states[0].frame_index)
        self.assertIn("gradient", states[0].extras)
        self.assertFalse(states[0].spectrum.requires_grad)
        self.assertNotEqual(states[0].extras["gradient"].data_ptr(), gradient.data_ptr())

    def test_live_renderer_can_save_selectable_panels_without_a_gui(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from fp.plotting import ReconstructionVisualizer
        model = model_fixture()
        data = simulate_measurements(torch.ones(model.grid.object_shape), model)
        with TemporaryDirectory() as temporary:
            visualizer = ReconstructionVisualizer(
                show=False, output_dir=temporary,
                views=("spectrum", "pupil_amplitude", "pupil_phase", "residual", "update", "convergence"),
            )
            try:
                SaviReconstructor(model, SolverOptions(1)).reconstruct(
                    data, observer=visualizer, observer_interval=3,
                )
                paths = sorted(Path(temporary).glob("*.png"))
                self.assertEqual(len(paths), 3)
                self.assertGreater(cv2.imread(str(paths[-1])).shape[0], 100)
                self.assertEqual(sorted(visualizer.metrics), [0, 1])
            finally:
                visualizer.close()

    def test_cli_debug_export_and_precision_options(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "0000.npy", np.ones((8, 8)))
            args = parse_args([
                str(directory), "-w", "79.99", "-t", "28", "-n", "1", "-r", "8", "-e", "1",
                "--device", "cpu", "--precision", "float32", "--debug-every", "2",
                "--debug-dir", str(directory / "debug"), "--debug-views", "spectrum", "pupil_amplitude",
            ])
            with redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(result.object_field.dtype, torch.complex64)
            self.assertEqual(len(list((directory / "debug").glob("*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
