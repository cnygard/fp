"""Opt-in native-resolution checks of locally available acquisition datasets."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from fp.cli import parse_args, run

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("FP_TEST_DATASETS") == "1", "set FP_TEST_DATASETS=1 to validate local acquisitions")
class RepresentativeDatasetTests(unittest.TestCase):
    def check_dataset(self, relative_path, config, shape):
        path = ROOT / relative_path
        if not path.exists():
            self.skipTest(f"local dataset unavailable: {path}")
        with TemporaryDirectory() as temporary:
            for algorithm in ("savi", "alternating_projection", "wirtinger_flow"):
                with self.subTest(algorithm=algorithm):
                    output = Path(temporary) / f"{algorithm}.npz"
                    args = parse_args([
                        str(path), "--config", str(ROOT / "configs" / config),
                        "--algorithm", algorithm, "--epochs", "3", "--output", str(output),
                        "--device", "cpu", "--precision", "float64",
                    ])
                    with redirect_stdout(io.StringIO()):
                        result = run(args)
                    initial, final = result.history[0].amplitude_rmse, result.history[-1].amplitude_rmse
                    self.assertTrue(torch.isfinite(result.object_field).all())
                    self.assertTrue(torch.isfinite(result.pupil).all())
                    self.assertEqual(result.object_field.shape, shape)
                    self.assertEqual(len(result.frame_ids), 225)
                    self.assertLess(final, initial)
                    with np.load(output, allow_pickle=False) as saved:
                        self.assertEqual(saved["object_field"].shape, shape)
                        self.assertEqual(len(saved["frame_ids"]), 225)
                        self.assertEqual(json.loads(saved["settings_json"].item())["algorithm"], algorithm)
                    print(f"{relative_path} | {algorithm} | amplitude RMSE {initial:.8g} -> {final:.8g}", flush=True)

    def test_usaf_hdr(self):
        self.check_dataset("Data/USAF_red.mat", "usaf_red.toml", (564, 564))

    def test_mousekidney_hdr(self):
        self.check_dataset("Data/MouseKidney_green.mat", "mousekidney_green.toml", (804, 804))

    def test_usaf_numbered_led(self):
        self.check_dataset("Data/USAF_red", "usaf_red_png.toml", (564, 564))

    def test_array_capture(self):
        self.check_dataset("Data/LED_sim/nonspeckle_incoherent_capture", "array_sim.toml", (600, 600))


if __name__ == "__main__":
    unittest.main()
