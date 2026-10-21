"""Human-readable resolved run settings, kept separate from numerical operators."""

from pathlib import Path
from platform import machine

import torch


def _shape(shape):
    return " x ".join(str(size) for size in shape)


def _dtype(dtype):
    return str(dtype).removeprefix("torch.")


def format_run_summary(args, model, measurements, options):
    """Describe effective physical sampling and actual compute settings before solving."""
    optics, grid = model.optics, model.grid
    if model.device.type == "cuda":
        device_name = torch.cuda.get_device_name(model.device)
    elif model.device.type == "mps":
        device_name = "Apple Metal GPU"
    else:
        device_name = f"CPU ({machine()})"
    aperture_description = (
        "equivalent aperture diameter" if args.objective_na is not None else "aperture diameter"
    )
    native_spacing = args.pixel_size_um * 1e-6
    object_spacing = optics.pixel_size_m / grid.upsample
    fov_h, fov_w = (size * optics.pixel_size_m for size in grid.measurement_shape)
    support_scale = 2.0 if args.algorithm == "savi" else 1.0
    supported = int(model.pupil_support_at_scale(support_scale).sum().item())
    pupil_pixels = model.pupil_support.numel()
    intensity_min = measurements.intensities.min().item()
    intensity_max = measurements.intensities.max().item()
    wavevectors = model.wavevectors_rad_per_m
    kx_min, ky_min = wavevectors.amin(dim=0).tolist()
    kx_max, ky_max = wavevectors.amax(dim=0).tolist()
    debug_enabled = args.debug or args.debug_dir is not None
    debug_summary = "off"
    if debug_enabled:
        destinations = []
        if args.debug:
            destinations.append("live")
        if args.debug_dir is not None:
            destinations.append(f"PNGs: {Path(args.debug_dir)}")
        debug_summary = (
            f"{', '.join(destinations)}; every {args.debug_every} updates + epoch boundaries; "
            f"views: {', '.join(args.debug_views)}"
        )
    calibrated = args.source in ("matlab", "led")
    geometry_description = (
        "calibrated LED illumination" if calibrated else f"spacing {args.translation:.6g} mm"
    )
    pupil_description = (
        "fixed pupil" if args.algorithm == "wirtinger_flow" else "joint recovery enabled"
    )
    solver_description = (
        "curvature-scaled backtracking; intensity loss; fixed pupil"
        if args.algorithm == "wirtinger_flow"
        else f"pupil learning rate {options.pupil_learning_rate:.6g}"
    )
    objective_description = (
        "  Objective: half mean-squared intensity error; pupil controls and projection stabilizers unused"
        if args.algorithm == "wirtinger_flow"
        else f"  Stabilizers: object tau1={options.object_epsilon:.6g}, pupil tau2={options.pupil_epsilon:.6g}"
    )
    return "\n".join(
        [
            "fp reconstruction",
            f"  Input: {Path(args.data_dir).resolve()} (source: {args.source})",
            f"  Measurements: {_shape(measurements.intensities.shape)} (frames x height x width), intensity [{intensity_min:.6g}, {intensity_max:.6g}]",
            (
                f"  Optics: wavelength {optics.wavelength_m * 1e9:.6g} nm; "
                f"{aperture_description} {optics.aperture_diameter_m * 1e3:.6g} mm; "
                f"object distance {optics.object_distance_m:.6g} m"
            ),
            (
                f"  Aperture model: circular, free-space; NA {optics.numerical_aperture:.6g}; "
                f"refractive index {optics.refractive_index:.6g}"
            ),
            (
                f"  Sampling: native {native_spacing * 1e6:.6g} um; "
                f"measurement {optics.pixel_size_m * 1e6:.6g} um; "
                f"reconstructed object {object_spacing * 1e6:.6g} um"
            ),
            f"  Field of view (height x width): {fov_h * 1e3:.6g} x {fov_w * 1e3:.6g} mm",
            (
                f"  Reconstruction grid: {_shape(grid.measurement_shape)} -> {_shape(grid.object_shape)}; "
                f"upsample {grid.upsample}x; field FFT scale 1/{grid.field_scale}"
            ),
            (
                f"  Geometry: {geometry_description}; "
                f"kx [{kx_min:.6g}, {kx_max:.6g}], ky [{ky_min:.6g}, {ky_max:.6g}] rad/m; "
                f"integer frequency patches"
            ),
            (
                f"  Pupil: {pupil_description}; "
                f"support radius {support_scale:g}x nominal, {supported}/{pupil_pixels} pixels ({100 * supported / pupil_pixels:.3g}%); "
                f"nominal aperture initialization"
            ),
            (
                f"  Solver: {args.algorithm}; epochs {options.epochs}; "
                f"object learning rate {options.learning_rate:.6g}; {solver_description}; "
                f"initialization: resized mean intensity"
            ),
            objective_description,
            (
                f"  Compute: {model.device} ({device_name}); "
                f"real {_dtype(model.real_dtype)}, complex {_dtype(model.complex_dtype)}; "
                f"requested device={model.requested_compute.device}, precision={model.requested_compute.precision}"
            ),
            f"  Torch: {torch.__version__}",
            f"  Debug: {debug_summary}",
            (
                f"  Result file: {args.output if args.output is not None else 'disabled'}; "
                f"final plot: {'on' if args.show else 'off'}"
            ),
        ]
    )
