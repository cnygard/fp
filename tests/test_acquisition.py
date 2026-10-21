"""Intensity-scale, sampling, calibration, and CLI acquisition contracts."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from numpy.testing import assert_allclose
from scipy.io import savemat
import torch

from fp.cli import parse_args, run
from fp.geometry import led_wavevectors
from fp.io import ArraySimLoader, LEDCalibration, MatlabHDRLoader


class LEDTests(unittest.TestCase):
    def test_no_slide_matches_direction_cosines_and_xy_sign(self):
        positions = np.array([[0., 0.], [.03, 0.], [0., -.04], [.03, -.04]])
        wavelength, height = 532e-9, .09
        expected = -2 * np.pi / wavelength * positions / np.sqrt(
            (positions**2).sum(axis=1)[:, None] + height**2)
        actual = led_wavevectors(positions, height, wavelength)
        assert_allclose(actual, expected, atol=1e-9, rtol=1e-14)
        # A direction cosine must not be interpreted as an angle and sin'ed again.
        self.assertLess(actual[1, 0], -3.7e6)

    def test_slide_satisfies_snell_ray_equation_and_central_led_is_zero(self):
        positions = np.array([[0., 0.], [.028, -.024], [-.004, .016]])
        height, thickness, index, wavelength = .09088, .001, 1.52, 630e-9
        k = led_wavevectors(positions, height, wavelength,
                           glass_thickness_m=thickness, glass_refractive_index=index).numpy()
        sine = np.linalg.norm(k, axis=1) / (2 * np.pi / wavelength)
        radial_distance = (height * sine / np.sqrt(1 - sine**2)
                           + thickness * (sine / index) / np.sqrt(1 - (sine / index)**2))
        assert_allclose(radial_distance, np.linalg.norm(positions, axis=1), atol=1e-16)
        assert_allclose(k[0], [0, 0])
        self.assertTrue(np.all(k[1] * positions[1] < 0))
        # n=1 is analytically a single uniform gap H+h.
        uniform = led_wavevectors(positions, height, wavelength,
                                 glass_thickness_m=thickness, glass_refractive_index=1.)
        assert_allclose(uniform, led_wavevectors(positions, height + thickness, wavelength), atol=1e-9)

    def test_mat_locations_keep_order_and_apply_calibration_in_xy(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "led.mat"
            savemat(path, {"xlocation_s": [[2, 0, -1]], "ylocation_s": [[0], [1], [-2]]})
            calibration = LEDCalibration(.004, .09, 0., offset_xy_m=(.001, -.002), rotation_deg=90)
            geometry = calibration.load(path, 532e-9)
            positions = np.array([[.001, .006], [-.003, -.002], [.009, -.006]])
            assert_allclose(geometry, led_wavevectors(positions, .09, 532e-9), atol=1e-9)
            savemat(path, {"wavevectors_rad_per_m": [[1, 2], [3, 4]]})
            assert_allclose(calibration.load(path, 532e-9), [[1, 2], [3, 4]])

    def test_invalid_calibration_reports_actionable_errors(self):
        for kwargs in ({"spacing_m": 0}, {"height_m": -1}, {"glass_thickness_m": -1},
                       {"glass_refractive_index": .5}, {"rotation_deg": float("nan")},
                       {"offset_xy_m": [0]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LEDCalibration(**kwargs)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "led.mat"
            for values in ({"xlocation_s": [[1, 2]]},
                           {"xlocation_s": [[1, 2]], "ylocation_s": [[1]]},
                           {"xlocation_s": np.ones((2, 2)), "ylocation_s": np.ones((2, 2))},
                           {"wavevectors_rad_per_m": [[float("nan"), 0]]},
                           {"wavevectors_rad_per_m": [[1e20, 0]]}):
                savemat(path, values)
                with self.subTest(values=values), self.assertRaises(ValueError):
                    LEDCalibration().load(path, 532e-9)


class MatlabTests(unittest.TestCase):
    def test_hdr_scale_frame_order_crop_and_effective_spacing(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.mat"
            cube = np.stack([np.full((4, 6), 2.75), np.full((4, 6), .25)], axis=2)
            savemat(path, {"imlow_HDR": cube})
            data = MatlabHDRLoader([[1, 0], [0, 2]], 1.845e-6, 2).load(path)
            assert_allclose(data.intensities[:, 0, 0], [2.75, .25])
            assert_allclose(data.wavevectors_rad_per_m, [[1, 0], [0, 2]])
            self.assertEqual(data.pixel_size_m, 3.69e-6)
            self.assertEqual(data.frame_ids, ("capture.mat:imlow_HDR[0]", "capture.mat:imlow_HDR[1]"))
            data = MatlabHDRLoader([[1, 0], [0, 2]], 1e-6, 4, intensity_scale=1/255).load(path)
            assert_allclose(data.intensities[:, 0, 0], np.array([2.75, .25])/255)

    def test_integer_mat_values_are_not_implicitly_normalized(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "integer.mat"
            savemat(path, {"custom": np.full((3, 3, 1), 1000, dtype=np.uint16)})
            data = MatlabHDRLoader([[0, 0]], 1e-6, 3, key="custom").load(path)
            assert_allclose(data.intensities, 1000)

    def test_missing_invalid_and_mismatched_cubes(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.mat"
            loader = MatlabHDRLoader([[0, 0]], 1e-6, 2)
            with self.assertRaises(FileNotFoundError):
                loader.load(path)
            for cube in (np.ones((2, 2)), np.ones((2, 2, 2)), -np.ones((2, 2, 1)),
                         np.full((2, 2, 1), np.nan), np.ones((2, 2, 1), dtype=complex)):
                savemat(path, {"imlow_HDR": cube})
                with self.subTest(shape=cube.shape), self.assertRaises(ValueError):
                    loader.load(path)
            savemat(path, {"wrong_key": np.ones((2, 2, 1))})
            with self.assertRaisesRegex(ValueError, "missing intensity variable"):
                loader.load(path)
            with patch("fp.io.matlab.loadmat", side_effect=NotImplementedError):
                with self.assertRaisesRegex(ValueError, "v7.3/HDF5"):
                    loader.load(path)

    def test_frame_patterns_and_missing_frames(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in range(2):
                cv2.imwrite(str(directory / f"ILED_{index:02d}.tif"), np.full((4, 4), index*1000, dtype=np.uint16))
            data = ArraySimLoader([[0, 0], [1, 0]], 1e-6, 4, ".tif", "ILED_{index:02d}").load(directory)
            self.assertEqual(data.frame_ids, ("ILED_00.tif", "ILED_01.tif"))
            assert_allclose(data.intensities[:, 0, 0], [0, 1000/65535])
            with self.assertRaises(FileNotFoundError):
                ArraySimLoader([[0, 0]], 1e-6, 4).load(directory)
        for pattern in ("../{index}", "constant", "{wrong}"):
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                ArraySimLoader([[0, 0], [1, 0]], 1e-6, frame_pattern=pattern)


class AcquisitionCliTests(unittest.TestCase):
    def test_mat_and_numbered_led_sources_with_all_algorithms_and_export(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            savemat(directory / "sample.mat", {"imlow_HDR": np.full((6, 6, 2), 2.25)})
            savemat(directory / "sample_LED_locations.mat", {"wavevectors_rad_per_m": [[0, 0], [100, 0]]})
            for index in range(2):
                np.save(directory / f"{index}.npy", np.full((6, 6), 2.25))
            (directory / "sample.toml").write_text(
                'objective_na = 0.1\nresolution = 6\nepochs = 1\n'
                'device = "cpu"\nprecision = "float64"\n')
            for algorithm in ("savi", "alternating_projection", "wirtinger_flow"):
                for source in ("matlab", "led"):
                    with self.subTest(source=source, algorithm=algorithm):
                        output = directory / f"{source}_{algorithm}.npz"
                        arguments = ([str(directory / "sample.mat")] if source == "matlab" else
                                     [temporary, "--source", "led", "--led-calibration", "sample_LED_locations.mat",
                                      "--objective-na", ".1", "-r", "6", "-e", "1", "--frame-pattern", "{index}"])
                        args = parse_args(arguments + ["--algorithm", algorithm, "--output", str(output)])
                        with redirect_stdout(io.StringIO()) as log:
                            result = run(args)
                        self.assertTrue(torch.isfinite(result.object_field).all())
                        self.assertIn("calibrated LED illumination", log.getvalue())
                        if algorithm == "wirtinger_flow":
                            self.assertIn("Pupil: fixed pupil", log.getvalue())
                            self.assertNotIn("joint recovery enabled", log.getvalue())
                        with np.load(output, allow_pickle=False) as saved:
                            settings = json.loads(saved["settings_json"].item())
                            self.assertEqual(settings["acquisition"]["source"], source)
                            self.assertEqual(len(saved["frame_ids"]), 2)

    def test_directory_mat_input_config_and_cli_override(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            savemat(directory / "capture.mat", {"imlow_HDR": np.ones((4, 4, 1))})
            savemat(directory / "led.mat", {"xlocation_s": [[0]], "ylocation_s": [[0]]})
            (directory / "config.toml").write_text(
                'source = "matlab"\ndata_name = "capture"\nled_calibration = "led.mat"\n'
                'objective_na = 0.1\nintensity_scale = 0.5\nresolution = 4\nepochs = 1\n')
            args = parse_args([temporary, "--intensity-scale", "2.0"])
            with redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(result.settings["acquisition"]["intensity_scale"], 2.0)
            self.assertEqual(result.object_field.shape, (8, 8))

    def test_explicit_config_and_optics_representation_overrides(self):
        with TemporaryDirectory() as temporary:
            config = Path(temporary) / "settings.toml"
            config.write_text('objective_na = 0.1\ntranslation = 1.0\n')
            args = parse_args([temporary, "--config", str(config), "--aperture_diameter", "20"])
            self.assertIsNone(args.objective_na)
            self.assertEqual(args.aperture_diameter, 20.)
            config.write_text('aperture_diameter = 20.0\ntranslation = 1.0\n')
            args = parse_args([temporary, "--config", str(config), "--objective-na", ".2"])
            self.assertIsNone(args.aperture_diameter)
            self.assertEqual(args.objective_na, .2)

    def test_led_requires_calibration_and_array_requires_translation(self):
        for arguments in (["--source", "led"], ["--source", "array_sim"], ["--source", "matlab"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([".", "--objective-na", ".1", *arguments])


if __name__ == "__main__":
    unittest.main()
