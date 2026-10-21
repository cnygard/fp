"""Numerical contracts and parity checks for the first migration increment."""

import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from minimal_fpm import (
    AlternatingProjectionReconstructor, FourierPtychographyModel, MeasurementSet,
    OpticalConfig, ReconstructionGrid, SaviReconstructor, SolverOptions,
)
from minimal_fpm.cli import parse_args, run
from minimal_fpm.fft import fft2c, ifft2c
from minimal_fpm.geometry import angles_to_wavevectors, array_wavevectors, spiral_order
from minimal_fpm.io import ArraySimLoader
from minimal_fpm.preprocessing import center_crop_resize
from minimal_fpm.results import save_result
from minimal_fpm.simulation import simulate_measurements

ROOT = Path(__file__).resolve().parents[1]


def make_model(shape=(8, 8), upsample=2):
    optics = OpticalConfig(1.85e-6, 532e-9, .6, 2.)
    height, width = shape
    dkx = 2 * np.pi / (width * optics.pixel_size_m)
    dky = 2 * np.pi / (height * optics.pixel_size_m)
    geometry = np.array([[0., 0.], [dkx, 0.], [0., -dky], [-dkx, dky]])
    return FourierPtychographyModel(optics, ReconstructionGrid(shape, upsample), geometry)


class MeasurementTests(unittest.TestCase):
    def test_reordering_keeps_images_geometry_and_ids_aligned(self):
        images = np.arange(12.).reshape(3, 2, 2)
        geometry = np.array([[0., 1.], [2., 3.], [4., 5.]])
        data = MeasurementSet(images, geometry, 1e-6, ("a", "b", "c"))
        images[:] = 0
        geometry[:] = 0
        reordered = data.reordered([2, 0, 1])
        assert_array_equal(reordered.intensities[:, 0, 0], [8, 0, 4])
        assert_array_equal(reordered.wavevectors_rad_per_m[:, 0], [4, 0, 2])
        self.assertEqual(reordered.frame_ids, ("c", "a", "b"))
        self.assertFalse(data.intensities.flags.writeable)
        self.assertFalse(data.wavevectors_rad_per_m.flags.writeable)
        with self.assertRaises(ValueError):
            data.reordered([0, 0, 1])

    def test_rejects_invalid_measurements(self):
        for images, geometry in [
            (np.ones((2, 3)), [[0, 0]]),
            (np.ones((2, 3, 3)), [[0, 0]]),
            (-np.ones((1, 3, 3)), [[0, 0]]),
            (np.full((1, 3, 3), np.nan), [[0, 0]]),
            (np.ones((1, 3, 3)), [[np.inf, 0]]),
            (np.ones((1, 3, 3), dtype=complex), [[0, 0]]),
        ]:
            with self.subTest(shape=images.shape, geometry=geometry):
                with self.assertRaises(ValueError):
                    MeasurementSet(images, geometry, 1e-6)

    def test_crop_resize_preserves_field_of_view(self):
        image = np.arange(48.).reshape(6, 8)
        cropped, spacing = center_crop_resize(image, 6, 2e-6)
        assert_array_equal(cropped, image[:, 1:7])
        resized, spacing = center_crop_resize(image, 3, 2e-6)
        self.assertEqual(resized.shape, (3, 3))
        self.assertEqual(spacing, 4e-6)
        portrait = image.T
        cropped, _ = center_crop_resize(portrait, 6, 2e-6)
        assert_array_equal(cropped, portrait[1:7, :])


class GeometryAndOpticsTests(unittest.TestCase):
    def test_spiral_is_complete_and_matches_legacy_for_supported_odd_arrays(self):
        for side in (1, 2, 3, 4, 15):
            order = spiral_order(side)
            assert_array_equal(np.sort(order), np.arange(side ** 2))
        assert_array_equal(spiral_order(3), [4, 5, 8, 7, 6, 3, 0, 1, 2])
        with self.assertRaises(ValueError):
            array_wavevectors(8, .01, 2., 532e-9)

    def test_angles_are_explicitly_radians_and_columns_are_xy(self):
        vectors = angles_to_wavevectors([[np.pi / 6, 0]], 500e-9)
        assert_allclose(vectors, [[2 * np.pi / 500e-9 * .5, 0]], atol=1e-8)

    def test_fft_shifts_only_spatial_axes_and_roundtrips(self):
        rng = np.random.default_rng(3)
        fields = rng.normal(size=(3, 5, 8)) + 1j * rng.normal(size=(3, 5, 8))
        transformed = fft2c(fields)
        for index in range(3):
            assert_allclose(transformed[index], np.fft.fftshift(np.fft.fft2(fields[index])))
        assert_allclose(ifft2c(transformed), fields, atol=1e-14)

    def test_constant_object_has_unit_intensity_for_odd_and_even_rectangular_grids(self):
        for shape in ((8, 8), (5, 7), (6, 9)):
            for scale in (1, 2, 3):
                model = FourierPtychographyModel(
                    make_model().optics, ReconstructionGrid(shape, scale), [[0, 0]],
                )
                data = model.predict_intensities(np.ones(model.grid.object_shape))
                assert_allclose(data, 1., atol=1e-14)

    def test_patch_sign_axes_and_bounds(self):
        model = make_model((6, 8))
        self.assertEqual(model.patches[1], (slice(3, 9), slice(3, 11)))
        self.assertEqual(model.patches[2], (slice(4, 10), slice(4, 12)))
        with self.assertRaisesRegex(ValueError, "frame 0"):
            FourierPtychographyModel(model.optics, model.grid, [[1e12, 0]])

    def test_forward_prediction_matches_independent_numpy_calculation(self):
        model = make_model()
        rng = np.random.default_rng(14)
        obj = rng.normal(size=model.grid.object_shape) + 1j * rng.normal(size=model.grid.object_shape)
        spectrum = np.fft.fftshift(np.fft.fft2(obj))
        expected = []
        for patch in model.patches:
            field = np.fft.ifft2(np.fft.ifftshift(spectrum[patch] * model.pupil)) / 4
            expected.append(np.abs(field) ** 2)
        assert_allclose(model.predict_intensities(obj), expected, rtol=1e-14, atol=1e-14)

    def test_rejects_geometry_and_sampling_mismatch(self):
        model = make_model()
        intensities = np.ones((4, 8, 8))
        for data in (
            MeasurementSet(intensities, model.wavevectors_rad_per_m[::-1], model.optics.pixel_size_m),
            MeasurementSet(intensities, model.wavevectors_rad_per_m, model.optics.pixel_size_m * 2),
        ):
            with self.assertRaises(ValueError):
                SaviReconstructor(model).reconstruct(data)

    def test_config_validation(self):
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                OpticalConfig(invalid, 532e-9, .1, 2.)
        for invalid in (0, 1.5, True):
            with self.assertRaises(ValueError):
                ReconstructionGrid((8, 8), invalid)


class ReconstructionTests(unittest.TestCase):
    def legacy_result(self, filename, observations, model, options):
        # Extract only numerical definitions: importing the legacy modules would
        # parse process arguments and read a dataset configuration as a side effect.
        namespace = {"np": np, "cv2": cv2, "tqdm": lambda values: values,
                     "args": SimpleNamespace(learning=options.learning_rate, debug=False, momentum=0)}
        util_tree = ast.parse((ROOT / "util.py").read_text())
        helpers = {"matlab_round", "F", "iF", "find_bounds", "kspace_parameterize"}
        nodes = [node for node in util_tree.body if isinstance(node, ast.FunctionDef) and node.name in helpers]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "legacy_helpers", "exec"), namespace)
        tree = ast.parse((ROOT / filename).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reconstruction")
        exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
        class MaskAdapter:
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return model.pupil.real.copy()
        config = {"N": model.grid.measurement_shape[0], "out_N": model.grid.object_shape[0],
                  "upsample": model.grid.upsample, "dx": model.optics.pixel_size_m,
                  "k": model.optics.wavenumber_rad_per_m, "M": MaskAdapter()}
        angles = np.arcsin(model.wavevectors_rad_per_m / config["k"])
        with redirect_stdout(io.StringIO()):
            return namespace["reconstruction"](observations.copy(), angles, options.epochs, config)

    def test_projection_algorithms_match_existing_numerical_loops(self):
        model = make_model()
        options = SolverOptions(3, .17)
        observations = np.random.default_rng(42).uniform(.01, 1, size=(4, 8, 8))
        measurements = MeasurementSet(observations, model.wavevectors_rad_per_m, model.optics.pixel_size_m)
        for solver, filename in (
            (SaviReconstructor, "savi_reconstruction.py"),
            (AlternatingProjectionReconstructor, "matlab_reconstruction.py"),
        ):
            with self.subTest(solver=solver.__name__):
                result = solver(model, options).reconstruct(measurements)
                expected = self.legacy_result(filename, observations, model, options)
                assert_allclose(result.object_field, expected, rtol=1e-12, atol=1e-12)
        assert_array_equal(observations, measurements.intensities)

    def test_reconstruction_reduces_synthetic_measurement_error_and_is_repeatable(self):
        model = make_model((16, 16))
        y, x = np.mgrid[-1:1:32j, -1:1:32j]
        truth = (.3 + .5 * np.exp(-5 * (x ** 2 + y ** 2))) * np.exp(.2j * x)
        measurements = simulate_measurements(truth, model)
        for solver in (AlternatingProjectionReconstructor, SaviReconstructor):
            with self.subTest(solver=solver.__name__):
                callbacks = []
                reconstructor = solver(model, SolverOptions(8, .3))
                result = reconstructor.reconstruct(measurements, callback=callbacks.append)
                self.assertLess(result.history[-1].amplitude_rmse, result.history[0].amplitude_rmse * .25)
                self.assertEqual([entry.epoch for entry in callbacks], list(range(1, 9)))
                assert_allclose(result.object_field, reconstructor.reconstruct(measurements).object_field)
                self.assertTrue(np.iscomplexobj(result.object_field))
                assert_array_equal(model.pupil, result.pupil)

    def test_lossless_result_export(self):
        model = make_model()
        data = simulate_measurements(np.ones(model.grid.object_shape), model)
        result = SaviReconstructor(model, SolverOptions(1)).reconstruct(data)
        with TemporaryDirectory() as temporary:
            path = save_result(result, Path(temporary) / "result.npz")
            with np.load(path, allow_pickle=False) as saved:
                assert_array_equal(saved["object_field"], result.object_field)
                assert_array_equal(saved["wavevectors_rad_per_m"], model.wavevectors_rad_per_m)
                self.assertEqual(json.loads(saved["settings_json"].item())["algorithm"], "savi")
                assert_array_equal(saved["frame_ids"], data.frame_ids)


class LoaderAndCliTests(unittest.TestCase):
    def test_npy_loader_preserves_intensity_scale_order_and_updates_sampling(self):
        geometry = np.array([[0., 0.], [1., 0.]])
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in range(2):
                np.save(directory / f"{index:04d}.npy", np.full((8, 12), index + .5))
            data = ArraySimLoader(geometry, 1e-6, resolution=4).load(directory)
            assert_allclose(data.intensities[:, 0, 0], [.5, 1.5])
            assert_array_equal(data.wavevectors_rad_per_m, geometry)
            self.assertEqual(data.pixel_size_m, 2e-6)
            self.assertEqual(data.frame_ids, ("0000.npy", "0001.npy"))
            with self.assertRaises(FileNotFoundError):
                ArraySimLoader(np.zeros((3, 2)), 1e-6, 4).load(directory)

    def test_raster_loader_preserves_uint16_dynamic_range(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertTrue(cv2.imwrite(str(directory / "0000.tif"), np.full((4, 4), 32768, dtype=np.uint16)))
            data = ArraySimLoader([[0, 0]], 1e-6, 4, ".tif").load(directory)
            assert_allclose(data.intensities, 32768 / 65535)

    def test_loader_rejects_mixed_fields_of_view(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "0000.npy", np.ones((4, 4)))
            np.save(directory / "0001.npy", np.ones((6, 6)))
            with self.assertRaisesRegex(ValueError, "fields of view"):
                ArraySimLoader([[0, 0], [1, 0]], 1e-6, 4).load(directory)

    def test_cli_toml_defaults_overrides_and_end_to_end_output(self):
        with TemporaryDirectory(prefix="fpm test ") as temporary:
            directory = Path(temporary)
            (directory / "config.toml").write_text(
                'source = "array_sim"\nmode = "traditional"\nsamples_count = 1\n'
                'aperture_diameter = 79.99\ntranslation = 28\nresolution = 8\nepochs = 3\n'
            )
            np.save(directory / "0000.npy", np.ones((8, 8)))
            output = directory / "result.npz"
            args = parse_args(["--epochs", "1", str(directory), "--output", str(output)])
            self.assertEqual(args.epochs, 1)
            self.assertEqual(args.samples_count, 1)
            with redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertTrue(output.is_file())
            self.assertEqual(result.object_field.shape, (16, 16))

    def test_cli_simulation_uses_the_same_pipeline(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertTrue(cv2.imwrite(str(directory / "truth.png"), np.full((12, 12, 3), 128, dtype=np.uint8)))
            args = parse_args([
                str(directory), "--source", "sim", "--data_name", "truth", "--file_extension", ".png",
                "-w", "79.99", "-t", "28", "-n", "1", "-r", "8", "-e", "1",
            ])
            with redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertLess(result.history[-1].amplitude_rmse, result.history[0].amplitude_rmse)

    def test_cli_help_and_unsupported_legacy_source(self):
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.toml").write_text('source = "umd_bench"\n')
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parse_args([str(directory)])
            self.assertEqual(caught.exception.code, 2)

    def test_import_has_no_cli_plotting_or_torch_side_effects(self):
        checked = subprocess.run([
            sys.executable, "-B", "-c",
            "import sys; sys.argv = ['host', '--unrelated']; import minimal_fpm; "
            "assert 'matplotlib.pyplot' not in sys.modules; "
            "assert 'torch' not in sys.modules; assert 'config' not in sys.modules",
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == "__main__":
    unittest.main()
