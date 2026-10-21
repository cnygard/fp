# fp

Fourier ptychography reconstruction. The first modular migration is available
under `src/fp` on this branch. The original root-level research scripts
remain available while additional acquisition types and algorithms are migrated.

## Run the migrated pipeline

Use the existing project environment, or run `uv sync` to create one.

```sh
uv run fp Data/LED_sim/nonspeckle_incoherent_capture --epochs 3
```

The dataset's flat `config.toml` supplies defaults; explicit command-line options
override them. Both `fp` and `python -m fp` use the new CLI.
Add `--show` to plot the amplitude and phase. To preserve the complex result,
pupil, measurement geometry, frame IDs, settings, and convergence metrics:

```sh
uv run fp Data/LED_sim/nonspeckle_incoherent_capture \
    --algorithm alternating_projection --epochs 3 --output result.npz
```

Choose `savi` (default) or `alternating_projection`. Nothing is saved or plotted
unless requested. Inspect all options with `uv run fp --help`.

The first increment supports single-wavelength circular-pupil array data
(`--source array_sim`) and synthetic acquisitions (`--source sim`). Bench slit,
rotating-slit, Pi, MATLAB, and Waller acquisition adapters, momentum, pupil
estimation, denoising, autograd, and multi-wavelength reconstruction still use the
legacy scripts. The migrated CLI rejects unsupported source/mode settings rather
than guessing how their geometry maps into this model.

For a synthetic acquisition, supply an amplitude image and the sampling
and optical parameters explicitly:

```sh
uv run fp Data --source sim --data_name usaf --file_extension .jpg \
    --aperture_diameter 79.99 --translation 28 --object_distance 2 \
    --samples_count 9 --resolution 64 --epochs 3
```

Source images are converted to grayscale while retaining integer bit depth.
These parameters are an execution example, not a calibration for that image.

## Python API

```python
from fp import (
    FourierPtychographyModel, OpticalConfig, ReconstructionGrid,
    SaviReconstructor, SolverOptions,
)
from fp.geometry import array_wavevectors
from fp.io import ArraySimLoader

geometry = array_wavevectors(
    count=225, translation_m=0.028, object_distance_m=2.0,
    wavelength_m=532e-9,
)
measurements = ArraySimLoader(
    wavevectors_rad_per_m=geometry,
    pixel_size_m=1.85e-6,  # native effective sample-plane spacing
    resolution=300,
).load("Data/LED_sim/nonspeckle_incoherent_capture")

optics = OpticalConfig(
    pixel_size_m=measurements.pixel_size_m,
    wavelength_m=532e-9,
    aperture_diameter_m=0.07999,
    object_distance_m=2.0,
)
model = FourierPtychographyModel(
    optics=optics,
    grid=ReconstructionGrid(measurements.intensities.shape[1:], upsample=2),
    wavevectors_rad_per_m=measurements.wavevectors_rad_per_m,
)
result = SaviReconstructor(model, SolverOptions(epochs=3)).reconstruct(measurements)
```

`simulate_measurements(object_field, model)` produces the same `MeasurementSet`.
Solvers accept an optional complex `initial_field` and an epoch metrics callback.
Result `amplitude` and `phase` are derived from the retained complex object field.

## Structure and conventions

- `config.py`: frozen physical, grid, and solver settings; no argument parsing.
- `data.py`: owned intensity arrays, wavevectors, effective sampling, and frame IDs.
- `io/`: source-specific decoding, with geometry supplied in acquisition order.
- `preprocessing.py`: center crop and resize, including effective pixel spacing.
- `geometry.py`: explicit acquisition-to-wavevector conversions and ordering.
- `fft.py`, `optics.py`: spatial-axis FFTs, pupil support, frequency patches,
  field/intensity prediction, and amplitude error evaluation.
- `reconstructors/`: algorithm implementations with one common protocol.
- `simulation.py`: synthetic measurements through the shared optical model.
- `results.py`, `plotting.py`: numerical export and optional visualization.
- `cli.py`: TOML/argument parsing and composition of these components.

Measurement arrays are **intensities**, shaped `(frames, height, width)`. Geometry
columns are `(kx, ky)` in **rad/m**, while image axes are `(y, x)`. Internal lengths
are meters. The CLI retains legacy mm aperture/translation options and named
wavelengths; `--pixel_size_um` specifies native spacing for loaded array images,
or measurement spacing for synthetic acquisitions. Cropping preserves spacing;
resizing preserves the cropped field of view and updates spacing. NPY data keeps
its scale; integer rasters use their dtype's full range, including 16-bit images.
This corrects legacy preprocessing when resizing changes resolution.

The model uses NumPy FFT normalization and centered spectra. A measured field is
`ifft2c(object_spectrum_patch * pupil) / upsample**2`; intensity is its squared
magnitude. Output pixel spacing is `measurement_pixel_size / upsample`.
Bounds are validated before reconstruction. The array adapter preserves the
legacy array angle/sign convention and assumes numbered files already follow
spiral acquisition order. Even-sized arrays use a complete spiral traversal;
the legacy traversal can leave entries uninitialized for those sizes.

`SaviReconstructor` deliberately preserves both active object corrections in the
current SAVI prototype, using the same residual for both. It does not activate
its commented-out pupil recovery or denoising. Alternating projection preserves
the MATLAB-style main loop without its separate momentum stage. Mean-intensity
initialization is retained for numerical comparison with both legacy loops.
`history` includes initial amplitude RMSE at epoch 0 and one value per epoch;
callbacks run after each completed epoch.

## Validation

```sh
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v
```

The tests compare both migrated solver loops against numerical functions
extracted from the legacy source without importing its global CLI. They also
check independent forward calculations, FFT scaling for odd/rectangular grids,
measurement/geometry alignment, preprocessing sampling, 16-bit image loading,
synthetic convergence, lossless export, CLI composition, and import side effects.

Next migration steps are additional acquisition adapters with explicitly stated
geometry, followed by Wirtinger/autograd solvers and algorithm changes evaluated
separately from this structural migration.
