"""Paper-equation references and joint object/pupil reconstruction checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from fp import (
    AlternatingProjectionReconstructor, FourierPtychographyModel, MeasurementSet,
    OpticalConfig, ReconstructionGrid, SaviReconstructor, SolverOptions, TorchConfig,
)
from fp.fft import fft2c
from fp.results import save_result


def fixture(shape=(8, 8), geometry_count=4):
    optics = OpticalConfig(1.85e-6, 532e-9, .15, 2.)
    dkx = 2 * np.pi / (shape[1] * optics.pixel_size_m)
    dky = 2 * np.pi / (shape[0] * optics.pixel_size_m)
    shifts = np.array([[0, 0], [dkx, 0], [0, dky], [-dkx, -dky]])[:geometry_count]
    return FourierPtychographyModel(
        optics, ReconstructionGrid(shape), shifts, compute=TorchConfig("cpu", "float64"),
    )


def numpy_reference(model, observations, initial_field, initial_pupil, options, algorithm):
    """Independent implementation of SAVI Eqs. 7-11 / normalized joint projection.

    Uses NumPy only and deliberately computes both corrections from saved old
    fields to detect aliasing, update-order, scaling, and normalization mistakes.
    """
    spectrum = np.fft.fftshift(np.fft.fft2(initial_field))
    pupil = initial_pupil.copy()
    scale = 2 if algorithm == "savi" else 1
    support = model.pupil_support_at_scale(scale).cpu().numpy()
    pupil *= support
    for _ in range(options.epochs):
        for index, patch in enumerate(model.patches):
            old_region, old_pupil = spectrum[patch].copy(), pupil.copy()
            before = old_region * old_pupil
            field = np.fft.ifft2(np.fft.ifftshift(before)) / model.grid.field_scale
            projection = np.sqrt(observations[index]) * np.exp(1j * np.angle(field))
            after = np.fft.fftshift(np.fft.fft2(projection)) * model.grid.field_scale
            delta = after - before
            if algorithm == "savi":
                object_change = (
                    np.abs(old_pupil) / max(np.abs(old_pupil).max(), options.object_epsilon)
                    * np.conj(old_pupil) / (np.abs(old_pupil) ** 2 + options.object_epsilon) * delta
                )
                pupil_change = (
                    np.abs(old_region) / max(np.abs(spectrum).max(), options.pupil_epsilon)
                    * np.conj(old_region) / (np.abs(old_region) ** 2 + options.pupil_epsilon) * delta
                )
            else:
                object_change = np.conj(old_pupil) * delta / max(np.abs(old_pupil).max() ** 2, options.object_epsilon)
                pupil_change = np.conj(old_region) * delta / max(np.abs(old_region).max() ** 2, options.pupil_epsilon)
            spectrum[patch] = old_region + options.learning_rate * object_change
            pupil = (old_pupil + options.pupil_learning_rate * pupil_change) * support
    return np.fft.ifft2(np.fft.ifftshift(spectrum)), pupil


class PupilRecoveryTests(unittest.TestCase):
    def test_savi_and_alternating_projection_match_independent_joint_equations(self):
        model = fixture()
        rng = np.random.default_rng(5)
        spectrum = rng.normal(size=model.grid.object_shape) + 1j * rng.normal(size=model.grid.object_shape)
        # An out-of-patch maximum distinguishes SAVI's GLOBAL object normalization
        # from accidentally normalizing by the current patch instead.
        spectrum[0, 0] = 50 + 2j
        initial_field = np.fft.ifft2(np.fft.ifftshift(spectrum))
        nominal = model.pupil.cpu().numpy()
        initial_pupil = nominal * (.7 + .5 * rng.random(nominal.shape)) * np.exp(.3j * rng.normal(size=nominal.shape))
        observations = rng.uniform(.001, .1, size=(4, *model.grid.measurement_shape))
        data = MeasurementSet(observations, model.wavevectors_rad_per_m, model.optics.pixel_size_m)
        options = SolverOptions(epochs=2, learning_rate=.6, pupil_learning_rate=.35,
                                object_epsilon=1e-6, pupil_epsilon=2e-6)
        for cls, algorithm in ((SaviReconstructor, "savi"), (AlternatingProjectionReconstructor, "alternating_projection")):
            with self.subTest(algorithm=algorithm):
                result = cls(model, options).reconstruct(data, initial_field=initial_field, initial_pupil=initial_pupil)
                expected_object, expected_pupil = numpy_reference(
                    model, observations, initial_field, initial_pupil, options, algorithm,
                )
                np.testing.assert_allclose(result.object_field.numpy(), expected_object, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(result.pupil.numpy(), expected_pupil, rtol=1e-11, atol=1e-12)

    def test_unit_relaxation_is_the_default_and_matches_savi_equations(self):
        options = SolverOptions(epochs=1)
        self.assertEqual(options.learning_rate, 1.)
        self.assertEqual(options.pupil_learning_rate, 1.)
        model = fixture(geometry_count=1)
        field = np.random.default_rng(6).normal(size=model.grid.object_shape).astype(complex)
        observations = np.full((1, 8, 8), .1)
        data = MeasurementSet(observations, model.wavevectors_rad_per_m, model.optics.pixel_size_m)
        expected_object, expected_pupil = numpy_reference(
            model, observations, field, model.pupil.numpy(), options, "savi",
        )
        result = SaviReconstructor(model, options).reconstruct(data, initial_field=field)
        np.testing.assert_allclose(result.object_field.numpy(), expected_object, atol=1e-12)
        np.testing.assert_allclose(result.pupil.numpy(), expected_pupil, atol=1e-12)

    def test_support_scale_and_pupil_growth_outside_nominal_aperture(self):
        model = fixture(geometry_count=1)
        expanded = model.pupil_support_at_scale(2.)
        self.assertGreater(expanded.sum(), model.pupil_support.sum())
        generator = torch.Generator().manual_seed(8)
        field = torch.randn(model.grid.object_shape, dtype=torch.complex128, generator=generator)
        true_pupil = model.pupil_support_at_scale(1.7).to(torch.complex128)
        observations = model.predict_intensities(field, pupil=true_pupil)
        data = MeasurementSet(observations, model.wavevectors_rad_per_m, model.optics.pixel_size_m)
        for cls, scale in ((SaviReconstructor, 2.), (AlternatingProjectionReconstructor, 1.)):
            with self.subTest(algorithm=cls.__name__):
                result = cls(model, SolverOptions(1)).reconstruct(data, initial_field=field)
                allowed = model.pupil_support_at_scale(scale)
                self.assertEqual(result.pupil[~allowed].abs().sum().item(), 0)
                self.assertTrue(result.settings["pupil_recovery"]["enabled"])
                self.assertEqual(result.settings["pupil_recovery"]["support_radius_scale"], scale)
                if scale == 2.:
                    self.assertGreater(result.pupil[expanded & ~model.pupil_support].abs().sum().item(), 1e-5)

    def test_both_solvers_fit_aberrated_measurements_and_change_the_pupil(self):
        model = fixture()
        generator = torch.Generator().manual_seed(10)
        field = torch.randn(model.grid.object_shape, dtype=torch.complex128, generator=generator)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 8, dtype=torch.float64),
                                torch.linspace(-1, 1, 8, dtype=torch.float64), indexing="ij")
        true_pupil = model.pupil * torch.polar(.8 + .1 * xx, .9 * (xx.square() + yy.square()) + .3 * xx)
        data = MeasurementSet(model.predict_intensities(field, pupil=true_pupil),
                              model.wavevectors_rad_per_m, model.optics.pixel_size_m)
        initial = model.pupil.clone()
        for cls in (SaviReconstructor, AlternatingProjectionReconstructor):
            with self.subTest(algorithm=cls.__name__):
                result = cls(model, SolverOptions(epochs=20, learning_rate=.3, pupil_learning_rate=.5)).reconstruct(
                    data, initial_field=field,
                )
                self.assertLess(result.history[-1].amplitude_rmse, result.history[0].amplitude_rmse * .5)
                self.assertGreater((result.pupil - initial).abs().sum().item(), .01)
                self.assertTrue(torch.isfinite(result.pupil).all())
                reported = model.amplitude_rmse(fft2c(result.object_field), data.intensities, pupil=result.pupil)
                self.assertAlmostEqual(reported, result.history[-1].amplitude_rmse, places=12)
        torch.testing.assert_close(model.pupil, initial)

    def test_state_export_and_restart_use_the_recovered_pupil(self):
        model = fixture()
        observations = torch.full((4, 8, 8), .2, dtype=torch.float64)
        data = MeasurementSet(observations, model.wavevectors_rad_per_m, model.optics.pixel_size_m)
        states = []
        solver = SaviReconstructor(model, SolverOptions(2))
        result = solver.reconstruct(data, observer=states.append)
        torch.testing.assert_close(states[-1].pupil, result.pupil)
        self.assertIn("pupil_update", states[-1].extras)
        self.assertGreater(sum(state.extras.get("pupil_update", torch.zeros(())).abs().sum().item() for state in states), 0)
        with TemporaryDirectory() as temporary:
            path = save_result(result, Path(temporary) / "result.npz")
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["pupil"], result.pupil.numpy())
        restarted = solver.reconstruct(data, initial_field=result.object_field, initial_pupil=result.pupil)
        self.assertAlmostEqual(restarted.history[0].amplitude_rmse, result.history[-1].amplitude_rmse, places=12)

    def test_zero_data_and_zero_initial_estimates_remain_finite(self):
        model = fixture()
        data = MeasurementSet(torch.zeros((4, 8, 8)), model.wavevectors_rad_per_m, model.optics.pixel_size_m)
        for cls in (SaviReconstructor, AlternatingProjectionReconstructor):
            with self.subTest(algorithm=cls.__name__):
                result = cls(model, SolverOptions(2)).reconstruct(data, initial_pupil=torch.zeros((8, 8)))
                self.assertTrue(torch.isfinite(result.object_field).all())
                self.assertTrue(torch.isfinite(result.pupil).all())
                self.assertEqual(result.history[-1].amplitude_rmse, 0)

    def test_initial_pupil_is_validated_and_owned(self):
        model = fixture()
        data = MeasurementSet(torch.ones((4, 8, 8)), model.wavevectors_rad_per_m, model.optics.pixel_size_m)
        solver = SaviReconstructor(model, SolverOptions(1))
        for pupil in (torch.ones((7, 8)), torch.full((8, 8), torch.nan)):
            with self.assertRaisesRegex(ValueError, "initial_pupil"):
                solver.reconstruct(data, initial_pupil=pupil)
        pupil = model.pupil.clone()
        expected = pupil.clone()
        solver.reconstruct(data, initial_pupil=pupil)
        torch.testing.assert_close(pupil, expected)
        torch.testing.assert_close(model.pupil, expected)

    def test_new_solver_controls_are_validated(self):
        for keyword in ("pupil_learning_rate", "object_epsilon", "pupil_epsilon"):
            for value in (0, -1, float("nan")):
                with self.subTest(keyword=keyword, value=value):
                    with self.assertRaises(ValueError):
                        SolverOptions(**{keyword: value})


if __name__ == "__main__":
    unittest.main()
