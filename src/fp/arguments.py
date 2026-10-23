"""Command-line options and dataset TOML defaults."""

import argparse
import tomllib
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Apply dataset defaults, then explicit command-line overrides."""
    parser = _build_parser()
    # Use the complete parser on both passes so options may precede data_dir.
    initial_args = parser.parse_args(argv)
    defaults = _config_defaults(initial_args)
    # Only declared CLI settings are consumed from the dataset configuration.
    known = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in defaults.items() if key in known})
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fp", description="Single-wavelength Fourier ptychography reconstruction"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--config", type=Path, help="explicit TOML settings file; CLI values override it"
    )

    inputs = parser.add_argument_group("input")
    inputs.add_argument(
        "-s", "--source", choices=("array_sim", "sim", "matlab", "led"), default="array_sim"
    )
    inputs.add_argument(
        "--data_name", help="image stem for sim or MATLAB filename stem when passing a directory"
    )
    inputs.add_argument("--file_extension", default=".npy")
    inputs.add_argument(
        "--frame-pattern",
        default="{index:04d}",
        help="numbered frame stem, e.g. {index} or ILED_{index:02d}",
    )
    inputs.add_argument(
        "--mat-key",
        default="imlow_HDR",
        help="MATLAB intensity cube variable (height,width,frames)",
    )
    inputs.add_argument(
        "--intensity-scale", type=float, default=1.0, help="explicit MATLAB intensity multiplier"
    )

    led = parser.add_argument_group("LED calibration")
    led.add_argument(
        "--led-calibration",
        type=Path,
        help="LED locations or calibrated wavevectors in a .mat file, relative to dataset",
    )
    led.add_argument("--led-spacing-mm", type=float, default=4.0)
    led.add_argument("--led-height-mm", type=float, default=90.88, help="LED-to-slide air gap")
    led.add_argument(
        "--glass-thickness-mm", type=float, default=1.0, help="use 0 to disable slide refraction"
    )
    led.add_argument("--glass-refractive-index", type=float, default=1.52)
    led.add_argument("--led-offset-x-mm", type=float, default=0.0)
    led.add_argument("--led-offset-y-mm", type=float, default=0.0)
    led.add_argument("--led-rotation-deg", type=float, default=0.0)

    optics = parser.add_argument_group("optics and sampling")
    optics.add_argument(
        "--objective-na",
        type=float,
        help="objective numerical aperture, instead of aperture diameter",
    )
    optics.add_argument("-n", "--samples_count", type=int, default=225)
    optics.add_argument("-r", "--resolution", type=int, default=300)
    optics.add_argument("-u", "--upsample", type=int, default=2)
    # optics.add_argument("--wavelength", choices=("red", "green", "blue"), default="green")
    optics.add_argument("--wavelength", type=float, default=532)
    optics.add_argument(
        "--pixel_size_um",
        type=float,
        default=1.85,
        help="native array pixel spacing; sim measurement spacing, in micrometers",
    )
    optics.add_argument("-w", "--aperture_diameter", type=float, help="aperture diameter in mm")
    optics.add_argument("-t", "--translation", type=float, help="array spacing in mm")
    optics.add_argument(
        "--object_distance", type=float, default=2.0, help="object distance in meters"
    )
    optics.add_argument("-m", "--mode", choices=("traditional",), default="traditional")

    solver = parser.add_argument_group("solver")
    solver.add_argument(
        "--algorithm", choices=("savi", "alternating_projection", "wirtinger_flow"), default="savi"
    )
    solver.add_argument("-e", "--epochs", type=int, default=10)
    solver.add_argument(
        "-l", "--learning", type=float, default=1.0, help="object-update relaxation"
    )
    solver.add_argument("--pupil-learning", type=float, default=1.0, help="pupil-update relaxation")
    solver.add_argument(
        "--object-epsilon",
        type=float,
        default=1e-10,
        help="object-update denominator stabilizer (SAVI tau1)",
    )
    solver.add_argument(
        "--pupil-epsilon",
        type=float,
        default=1e-10,
        help="pupil-update denominator stabilizer (SAVI tau2)",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        type=Path,
        help="save complex object, pupil, metrics, and settings as .npz",
    )
    output.add_argument("--show", action="store_true", help="display amplitude and phase")

    compute = parser.add_argument_group("compute")
    compute.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda[:index], mps, or auto (default: prefer CUDA, MPS, CPU)",
    )
    compute.add_argument(
        "--precision",
        choices=("auto", "float32", "float64"),
        default="auto",
        help="auto selects the highest supported precision on the selected device",
    )

    debug = parser.add_argument_group("diagnostics")
    debug.add_argument(
        "-d", "--debug", action="store_true", help="show live reconstruction updates"
    )
    debug.add_argument(
        "--debug-every",
        type=int,
        default=10,
        help="updates between debug snapshots; epoch ends are always included",
    )
    debug.add_argument(
        "--debug-dir", type=Path, help="save debug PNG snapshots; works without a GUI"
    )
    from .plotting import AVAILABLE_VIEWS, DEFAULT_VIEWS

    debug.add_argument("--debug-views", nargs="+", choices=AVAILABLE_VIEWS, default=DEFAULT_VIEWS)
    return parser


def _config_defaults(args: argparse.Namespace) -> dict:
    config_path = (
        args.data_dir.with_suffix(".toml")
        if args.data_dir.suffix.lower() == ".mat"
        else args.data_dir / "config.toml"
    )
    if args.config is not None:
        config_path = args.config
    defaults = {"source": "matlab"} if args.data_dir.suffix.lower() == ".mat" else {}
    if args.config is not None or config_path.is_file():
        with config_path.open("rb") as stream:
            defaults.update(tomllib.load(stream))
    # Explicit optics options override the alternative representation in TOML.
    if args.aperture_diameter is not None and args.objective_na is None:
        defaults.pop("objective_na", None)
    if args.objective_na is not None and args.aperture_diameter is None:
        defaults.pop("aperture_diameter", None)
    return defaults


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    # argparse choices do not validate defaults read from a TOML file.
    for action in parser._actions:
        if action.choices:
            value = getattr(args, action.dest)
            values = value if action.nargs in ("+", "*") else (value,)
            if any(item not in action.choices for item in values):
                parser.error(f"unsupported {action.dest}: {value!r}; supported: {action.choices}")
    if args.aperture_diameter is None and args.objective_na is None:
        parser.error(
            "--aperture_diameter or --objective-na is required (or set it in configuration)"
        )
    if args.aperture_diameter is not None and args.objective_na is not None:
        parser.error("choose either --aperture_diameter or --objective-na")
    if args.source in ("array_sim", "sim") and args.translation is None:
        parser.error("--translation is required for array geometry")
    if args.source == "matlab" and args.data_dir.suffix.lower() != ".mat" and not args.data_name:
        parser.error("--source matlab requires a .mat input path or --data_name")
    if args.source == "led" and args.led_calibration is None:
        parser.error("--source led requires --led-calibration")
    if args.source == "sim" and not args.data_name:
        parser.error("--source sim requires --data_name (image filename without its extension)")
