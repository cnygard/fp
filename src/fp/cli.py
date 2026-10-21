"""Command-line configuration, acquisition, reconstruction, and output."""

import argparse
import tomllib
from dataclasses import asdict, replace
from math import asin, tan
from pathlib import Path

from .config import OpticalConfig, ReconstructionGrid, SolverOptions, _positive, _positive_integer
from .geometry import array_wavevectors
from .io import ArraySimLoader, LEDCalibration, MatlabHDRLoader
from .optics import FourierPtychographyModel
from .preprocessing import resize_image
from .reconstructors import AlternatingProjectionReconstructor, SaviReconstructor, WirtingerFlowReconstructor
from .results import save_result
from .reporting import format_run_summary
from .tensors import TorchConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="fp", description="Single-wavelength Fourier ptychography reconstruction"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--config", type=Path, help="explicit TOML settings file; CLI values override it")
    parser.add_argument(
        "-s", "--source", choices=("array_sim", "sim", "matlab", "led"), default="array_sim"
    )
    parser.add_argument(
        "--algorithm", choices=("savi", "alternating_projection", "wirtinger_flow"), default="savi"
    )
    parser.add_argument("--data_name", help="image stem for sim or MATLAB filename stem when passing a directory")
    parser.add_argument("--file_extension", default=".npy")
    parser.add_argument("--frame-pattern", default="{index:04d}", help="numbered frame stem, e.g. {index} or ILED_{index:02d}")
    parser.add_argument("--mat-key", default="imlow_HDR", help="MATLAB intensity cube variable (height,width,frames)")
    parser.add_argument("--intensity-scale", type=float, default=1.0, help="explicit MATLAB intensity multiplier")
    parser.add_argument("--led-calibration", type=Path, help="LED locations or calibrated wavevectors in a .mat file, relative to dataset")
    parser.add_argument("--led-spacing-mm", type=float, default=4.0)
    parser.add_argument("--led-height-mm", type=float, default=90.88, help="LED-to-slide air gap")
    parser.add_argument("--glass-thickness-mm", type=float, default=1.0, help="use 0 to disable slide refraction")
    parser.add_argument("--glass-refractive-index", type=float, default=1.52)
    parser.add_argument("--led-offset-x-mm", type=float, default=0.0)
    parser.add_argument("--led-offset-y-mm", type=float, default=0.0)
    parser.add_argument("--led-rotation-deg", type=float, default=0.0)
    parser.add_argument("--objective-na", type=float, help="objective numerical aperture, instead of aperture diameter")
    parser.add_argument("-n", "--samples_count", type=int, default=225)
    parser.add_argument("-r", "--resolution", type=int, default=300)
    parser.add_argument("-u", "--upsample", type=int, default=2)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-l", "--learning", type=float, default=1.0, help="object-update relaxation")
    parser.add_argument("--pupil-learning", type=float, default=1.0, help="pupil-update relaxation")
    parser.add_argument("--object-epsilon", type=float, default=1e-10, help="object-update denominator stabilizer (SAVI tau1)")
    parser.add_argument("--pupil-epsilon", type=float, default=1e-10, help="pupil-update denominator stabilizer (SAVI tau2)")
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
    config_path = (initial_args.data_dir.with_suffix(".toml")
                   if initial_args.data_dir.suffix.lower() == ".mat"
                   else initial_args.data_dir / "config.toml")
    if initial_args.config is not None:
        config_path = initial_args.config
    defaults = {"source": "matlab"} if initial_args.data_dir.suffix.lower() == ".mat" else {}
    if initial_args.config is not None or config_path.is_file():
        with config_path.open("rb") as stream:
            defaults.update(tomllib.load(stream))
    # Explicit optics options override the alternative representation in TOML.
    if initial_args.aperture_diameter is not None and initial_args.objective_na is None:
        defaults.pop("objective_na", None)
    if initial_args.objective_na is not None and initial_args.aperture_diameter is None:
        defaults.pop("aperture_diameter", None)
    # Only declared CLI settings are consumed from the dataset configuration.
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
    if args.aperture_diameter is None and args.objective_na is None:
        parser.error("--aperture_diameter or --objective-na is required (or set it in configuration)")
    if args.aperture_diameter is not None and args.objective_na is not None:
        parser.error("choose either --aperture_diameter or --objective-na")
    if args.source in ("array_sim", "sim") and args.translation is None:
        parser.error("--translation is required for array geometry")
    if args.source == "matlab" and args.data_dir.suffix.lower() != ".mat" and not args.data_name:
        parser.error("--source matlab requires a .mat input path or --data_name")
    if args.source == "led" and args.led_calibration is None:
        parser.error("--source led requires --led-calibration")
    if args.source == "sim" and not args.data_name:
        parser.error(
            "--source sim requires --data_name (image filename without its extension)"
        )
    return args


def run(args):
    compute = TorchConfig(device=args.device, precision=args.precision)
    _positive_integer("debug_every", args.debug_every)
    aperture_m = args.aperture_diameter * 1e-3 if args.aperture_diameter is not None else None
    if args.objective_na is not None:
        _positive("objective_na", args.objective_na)
        if args.objective_na >= 1.0003:
            raise ValueError("objective_na must be below the sample refractive index (1.0003)")
        aperture_m = 2 * args.object_distance * tan(asin(args.objective_na / 1.0003))
    optics = OpticalConfig(
        pixel_size_m=args.pixel_size_um * 1e-6,
        wavelength_m={"red": 630e-9, "green": 532e-9, "blue": 475e-9}[args.wavelength],
        aperture_diameter_m=aperture_m,
        object_distance_m=args.object_distance,
    )
    grid = ReconstructionGrid((args.resolution, args.resolution), args.upsample)
    options = SolverOptions(
        epochs=args.epochs, learning_rate=args.learning, pupil_learning_rate=args.pupil_learning,
        object_epsilon=args.object_epsilon, pupil_epsilon=args.pupil_epsilon,
    )
    acquisition = {"source": args.source, "input": str(args.data_dir.resolve())}
    if args.source in ("matlab", "led"):
        base = args.data_dir.parent if args.data_dir.suffix.lower() == ".mat" else args.data_dir
        mat_path = args.data_dir if args.data_dir.suffix.lower() == ".mat" else base / f"{args.data_name}.mat"
        calibration_path = args.led_calibration
        if calibration_path is None:
            calibration_path = mat_path.with_name(f"{mat_path.stem}_LED_locations.mat")
        else:
            calibration_path = Path(calibration_path)
            if not calibration_path.is_absolute():
                calibration_path = base / calibration_path
        calibration = LEDCalibration(
            spacing_m=args.led_spacing_mm * 1e-3, height_m=args.led_height_mm * 1e-3,
            glass_thickness_m=args.glass_thickness_mm * 1e-3,
            glass_refractive_index=args.glass_refractive_index,
            offset_xy_m=(args.led_offset_x_mm * 1e-3, args.led_offset_y_mm * 1e-3),
            rotation_deg=args.led_rotation_deg,
        )
        geometry = calibration.load(calibration_path, optics.wavelength_m)
        acquisition.update(led_calibration=str(calibration_path.resolve()), led=asdict(calibration))
        if args.source == "matlab":
            measurements = MatlabHDRLoader(
                geometry, optics.pixel_size_m, args.resolution, args.mat_key, args.intensity_scale,
            ).load(mat_path)
            acquisition.update(mat_key=args.mat_key, intensity_scale=args.intensity_scale)
        else:
            measurements = ArraySimLoader(
                geometry, optics.pixel_size_m, args.resolution, args.file_extension, args.frame_pattern,
            ).load(base)
            acquisition.update(frame_pattern=args.frame_pattern, file_extension=args.file_extension)
    else:
        geometry = array_wavevectors(
            args.samples_count, args.translation * 1e-3, optics.object_distance_m, optics.wavelength_m,
        )
        if args.source == "array_sim":
            measurements = ArraySimLoader(
                geometry, optics.pixel_size_m, args.resolution, args.file_extension, args.frame_pattern,
            ).load(args.data_dir)
            acquisition.update(frame_pattern=args.frame_pattern, file_extension=args.file_extension)
    if args.source != "sim":
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
        "wirtinger_flow": WirtingerFlowReconstructor,
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
    result.settings["acquisition"] = acquisition
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
