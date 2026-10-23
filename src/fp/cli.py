"""Acquire measurements, run a reconstruction, and present its results."""

from argparse import Namespace
from dataclasses import asdict, replace
from math import asin, tan
from pathlib import Path

from .arguments import parse_args as parse_args
from .config import OpticalConfig, ReconstructionGrid, SolverOptions
from .data import MeasurementSet
from .geometry import array_wavevectors
from .io import ArraySimLoader, LEDCalibration, MatlabHDRLoader
from .optics import FourierPtychographyModel
from .preprocessing import resize_image
from .reconstructors import (
    AlternatingProjectionReconstructor,
    SaviReconstructor,
    WirtingerFlowReconstructor,
)
from .reporting import format_run_summary
from .results import ReconstructionResult, save_result
from .tensors import TorchConfig
from .validation import require_positive, require_positive_integer


def _optical_config(args: Namespace) -> OpticalConfig:
    aperture_m = args.aperture_diameter * 1e-3 if args.aperture_diameter is not None else None
    if args.objective_na is not None:
        require_positive("objective_na", args.objective_na)
        if args.objective_na >= 1.0003:
            raise ValueError("objective_na must be below the sample refractive index (1.0003)")
        # Double the pupil diameter to account for alignment error
        aperture_m = 2 * args.object_distance * tan(asin(args.objective_na / 1.0003))
    return OpticalConfig(
        pixel_size_m=args.pixel_size_um * 1e-6,
        # wavelength_m={"red": 630e-9, "green": 532e-9, "blue": 475e-9}[args.wavelength],
        wavelength_m=args.wavelength*1e-9,
        aperture_diameter_m=aperture_m,
        object_distance_m=args.object_distance,
    )


def _load_calibrated(args: Namespace, optics: OpticalConfig) -> tuple[MeasurementSet, dict]:
    """Load calibrated LED frames in their supplied acquisition order."""
    base = args.data_dir.parent if args.data_dir.suffix.lower() == ".mat" else args.data_dir
    mat_path = (
        args.data_dir if args.data_dir.suffix.lower() == ".mat" else base / f"{args.data_name}.mat"
    )
    if args.led_calibration is None:
        calibration_path = mat_path.with_name(f"{mat_path.stem}_LED_locations.mat")
    else:
        calibration_path = Path(args.led_calibration)
        if not calibration_path.is_absolute():
            calibration_path = base / calibration_path
    calibration = LEDCalibration(
        spacing_m=args.led_spacing_mm * 1e-3,
        height_m=args.led_height_mm * 1e-3,
        glass_thickness_m=args.glass_thickness_mm * 1e-3,
        glass_refractive_index=args.glass_refractive_index,
        offset_xy_m=(args.led_offset_x_mm * 1e-3, args.led_offset_y_mm * 1e-3),
        rotation_deg=args.led_rotation_deg,
    )
    geometry = calibration.load(calibration_path, optics.wavelength_m)
    metadata = {"led_calibration": str(calibration_path.resolve()), "led": asdict(calibration)}
    if args.source == "matlab":
        measurements = MatlabHDRLoader(
            geometry,
            optics.pixel_size_m,
            resolution=args.resolution,
            key=args.mat_key,
            intensity_scale=args.intensity_scale,
        ).load(mat_path)
        metadata.update(mat_key=args.mat_key, intensity_scale=args.intensity_scale)
    else:
        measurements = _load_numbered(args, optics, geometry, base)
        metadata.update(frame_pattern=args.frame_pattern, file_extension=args.file_extension)
    return measurements, metadata


def _load_numbered(args, optics, geometry, directory) -> MeasurementSet:
    return ArraySimLoader(
        geometry,
        optics.pixel_size_m,
        resolution=args.resolution,
        file_extension=args.file_extension,
        frame_pattern=args.frame_pattern,
    ).load(directory)


def _simulate(args, model: FourierPtychographyModel) -> MeasurementSet:
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
    object_field = resize_image(image, model.grid.object_shape)
    return simulate_measurements(object_field, model)


def _prepare_acquisition(
    args, optics, grid, compute
) -> tuple[FourierPtychographyModel, MeasurementSet, dict]:
    metadata = {"source": args.source, "input": str(args.data_dir.resolve())}
    if args.source in ("matlab", "led"):
        measurements, calibration = _load_calibrated(args, optics)
        geometry = measurements.wavevectors_rad_per_m
        metadata.update(calibration)
    else:
        geometry = array_wavevectors(
            args.samples_count,
            args.translation * 1e-3,
            optics.object_distance_m,
            optics.wavelength_m,
        )
        if args.source == "sim":
            model = FourierPtychographyModel(optics, grid, geometry, compute=compute)
            return model, _simulate(args, model), metadata
        measurements = _load_numbered(args, optics, geometry, args.data_dir)
        metadata.update(frame_pattern=args.frame_pattern, file_extension=args.file_extension)
    # Cropping and resizing change the measurement spacing, preserving the field of view.
    optics = replace(optics, pixel_size_m=measurements.pixel_size_m)
    model = FourierPtychographyModel(optics, grid, geometry, compute=compute)
    return model, measurements, metadata


def _reconstruct(args, model, measurements, options) -> ReconstructionResult:
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
    return result


def run(args: Namespace) -> ReconstructionResult:
    compute = TorchConfig(device=args.device, precision=args.precision)
    require_positive_integer("debug_every", args.debug_every)
    optics = _optical_config(args)
    grid = ReconstructionGrid((args.resolution, args.resolution), args.upsample)
    options = SolverOptions(
        epochs=args.epochs,
        learning_rate=args.learning,
        pupil_learning_rate=args.pupil_learning,
        object_epsilon=args.object_epsilon,
        pupil_epsilon=args.pupil_epsilon,
    )
    model, measurements, acquisition = _prepare_acquisition(args, optics, grid, compute)
    print(format_run_summary(args, model, measurements, options), flush=True)
    result = _reconstruct(args, model, measurements, options)
    result.settings["acquisition"] = acquisition
    if args.output:
        print(f"Saved {save_result(result, args.output)}")
    if args.show:
        from .plotting import plot_result

        plot_result(result)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, TypeError, RuntimeError) as error:
        raise SystemExit(f"fp: {error}") from error
