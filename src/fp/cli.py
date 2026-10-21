"""CLI composition boundary for the first migrated acquisition paths."""

import argparse
import tomllib
from dataclasses import replace
from pathlib import Path

from .config import OpticalConfig, ReconstructionGrid, SolverOptions, _positive_integer
from .geometry import array_wavevectors
from .io import ArraySimLoader
from .optics import FourierPtychographyModel
from .preprocessing import resize_image
from .reconstructors import AlternatingProjectionReconstructor, SaviReconstructor
from .results import save_result
from .reporting import format_run_summary
from .tensors import TorchConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="fp", description="Modular single-wavelength array FPM reconstruction"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "-s", "--source", choices=("array_sim", "sim"), default="array_sim"
    )
    parser.add_argument(
        "--algorithm", choices=("savi", "alternating_projection"), default="savi"
    )
    parser.add_argument("--data_name", help="ground-truth image stem for --source sim")
    parser.add_argument("--file_extension", default=".npy")
    parser.add_argument("-n", "--samples_count", type=int, default=225)
    parser.add_argument("-r", "--resolution", type=int, default=300)
    parser.add_argument("-u", "--upsample", type=int, default=2)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-l", "--learning", type=float, default=0.1)
    parser.add_argument(
        "--wavelength", choices=("red", "green", "blue"), default="green"
    )
    parser.add_argument(
        "--pixel_size_um",
        type=float,
        default=1.85,
        help="native array pixel spacing; sim measurement spacing, in micrometers",
    )
    parser.add_argument(
        "-w", "--aperture_diameter", type=float, help="aperture diameter in mm"
    )
    parser.add_argument("-t", "--translation", type=float, help="array spacing in mm")
    parser.add_argument(
        "--object_distance", type=float, default=2.0, help="object distance in meters"
    )
    parser.add_argument("-m", "--mode", choices=("traditional",), default="traditional")
    parser.add_argument(
        "--output",
        type=Path,
        help="save complex object, pupil, metrics, and settings as .npz",
    )
    parser.add_argument(
        "--show", action="store_true", help="display amplitude and phase"
    )
    parser.add_argument(
        "--device", default="auto", help="cpu, cuda[:index], mps, or auto (default: prefer CUDA, MPS, CPU)"
    )
    parser.add_argument(
        "--precision", choices=("auto", "float32", "float64"), default="auto",
        help="auto selects the highest supported precision on the selected device"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="show live reconstruction updates"
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=10,
        help="updates between debug snapshots; epoch ends are always included",
    )
    parser.add_argument(
        "--debug-dir", type=Path, help="save debug PNG snapshots; works without a GUI"
    )
    from .plotting import AVAILABLE_VIEWS, DEFAULT_VIEWS

    parser.add_argument(
        "--debug-views", nargs="+", choices=AVAILABLE_VIEWS, default=DEFAULT_VIEWS
    )
    # Use the complete parser on both passes so options may precede data_dir.
    initial_args = parser.parse_args(argv)
    config_path = initial_args.data_dir / "config.toml"
    defaults = {}
    if config_path.is_file():
        with config_path.open("rb") as stream:
            defaults = tomllib.load(stream)
    # Ignore unrelated legacy TOML settings; only declared CLI settings are consumed.
    known = {action.dest for action in parser._actions}
    parser.set_defaults(
        **{key: value for key, value in defaults.items() if key in known}
    )
    args = parser.parse_args(argv)
    # argparse choices do not validate defaults read from a TOML file.
    for action in parser._actions:
        if action.choices:
            value = getattr(args, action.dest)
            values = value if action.nargs in ("+", "*") else (value,)
            if any(item not in action.choices for item in values):
                parser.error(
                    f"unsupported {action.dest}: {value!r}; supported: {action.choices}"
                )
    if args.aperture_diameter is None or args.translation is None:
        parser.error(
            "--aperture_diameter and --translation are required (or set them in config.toml)"
        )
    if args.source == "sim" and not args.data_name:
        parser.error(
            "--source sim requires --data_name (image filename without its extension)"
        )
    return args


def run(args):
    compute = TorchConfig(device=args.device, precision=args.precision)
    _positive_integer("debug_every", args.debug_every)
    optics = OpticalConfig(
        pixel_size_m=args.pixel_size_um * 1e-6,
        wavelength_m={"red": 630e-9, "green": 532e-9, "blue": 475e-9}[args.wavelength],
        aperture_diameter_m=args.aperture_diameter * 1e-3,
        object_distance_m=args.object_distance,
    )
    grid = ReconstructionGrid((args.resolution, args.resolution), args.upsample)
    options = SolverOptions(args.epochs, args.learning)
    geometry = array_wavevectors(
        args.samples_count,
        args.translation * 1e-3,
        optics.object_distance_m,
        optics.wavelength_m,
    )
    if args.source == "array_sim":
        measurements = ArraySimLoader(
            geometry, optics.pixel_size_m, args.resolution, args.file_extension
        ).load(args.data_dir)
        optics = replace(optics, pixel_size_m=measurements.pixel_size_m)
        model = FourierPtychographyModel(optics, grid, geometry, compute=compute)
    else:
        import cv2
        import numpy as np

        from .simulation import simulate_measurements

        path = args.data_dir / f"{args.data_name}{args.file_extension}"
        if not path.is_file():
            raise FileNotFoundError(path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if image is None or image.ndim != 2:
            raise ValueError(f"{path}: could not decode a ground-truth amplitude image")
        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float64) / np.iinfo(image.dtype).max
        height, width = grid.object_shape
        object_field = resize_image(image, (height, width))
        model = FourierPtychographyModel(optics, grid, geometry, compute=compute)
        measurements = simulate_measurements(object_field, model)
    print(format_run_summary(args, model, measurements, options), flush=True)
    solver_type = {
        "savi": SaviReconstructor,
        "alternating_projection": AlternatingProjectionReconstructor,
    }[args.algorithm]

    def progress(metrics):
        print(
            f"Epoch {metrics.epoch}/{options.epochs}: amplitude RMSE {metrics.amplitude_rmse:.6g}"
        )

    visualizer = None
    if args.debug or args.debug_dir is not None:
        from .plotting import ReconstructionVisualizer

        visualizer = ReconstructionVisualizer(
            views=args.debug_views,
            show=args.debug,
            output_dir=args.debug_dir,
        )
    try:
        result = solver_type(model, options).reconstruct(
            measurements,
            callback=progress,
            observer=visualizer,
            observer_interval=args.debug_every,
        )
    except BaseException:
        if visualizer is not None:
            visualizer.close()
        raise
    if visualizer is not None:
        visualizer.finish()
        if not args.debug:
            visualizer.close()
    if args.output:
        print(f"Saved {save_result(result, args.output)}")
    if args.show:
        from .plotting import plot_result

        plot_result(result)
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, TypeError, RuntimeError) as error:
        raise SystemExit(f"fp: {error}") from error
