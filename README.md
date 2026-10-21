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
    SaviReconstructor, SolverOptions, TorchConfig,
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
    compute=TorchConfig(),  # prefer an accelerator and its highest supported precision
)
result = SaviReconstructor(model, SolverOptions(epochs=3)).reconstruct(measurements)
```

`simulate_measurements(object_field, model)` produces the same `MeasurementSet`.
Solvers accept an optional complex `initial_field` and an epoch metrics callback.
Result `amplitude` and `phase` are derived from the retained complex object field.

## Structure and conventions

- `config.py`: frozen physical, grid, and solver settings; no argument parsing.
- `tensors.py`: runtime-probed device and precision selection.
- `reporting.py`: resolved physical, numerical, and runtime settings for CLI runs.
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

Measurement tensors are **intensities**, shaped `(frames, height, width)`. Geometry
columns are `(kx, ky)` in **rad/m**, while image axes are `(y, x)`. Internal lengths
are meters. The CLI retains legacy mm aperture/translation options and named
wavelengths; `--pixel_size_um` specifies native spacing for loaded array images,
or measurement spacing for synthetic acquisitions. Cropping preserves spacing;
resizing preserves the cropped field of view and updates spacing. NPY data keeps
its scale; integer rasters use their dtype's full range, including 16-bit images.
This corrects legacy preprocessing when resizing changes resolution.

The model uses Torch FFTs with default backward normalization and centered spectra. A measured field is
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
Torch-specific checks verify the field adjoint, complex autograd for the object
and pupil, precision/device consistency, snapshot isolation, and debug rendering.

## Torch computation

Data, preprocessing, geometry, FFTs, forward prediction, solver updates, and result
arrays now use Torch tensors. NumPy remains at NPY/NPZ file boundaries; OpenCV
only decodes images. Plotting explicitly detaches tensors and transfers displayed
arrays to CPU. No NumPy conversion or CPU image resize occurs in the solver loop.

Both device and precision default to `auto`, in the CLI and Python API.
Resolution tries available CUDA, then MPS, then CPU. On each device it tries
float64 / complex128 first, then float32 / complex64, testing forward and inverse
complex FFTs in the installed Torch runtime. This prefers an accelerator and its
highest supported precision; it does not benchmark which device is fastest for
a particular image size. A typical CUDA or CPU selection is float64 / complex128;
a typical MPS selection is float32 / complex64.

Use `--device cpu`, `--device cuda`, `--device cuda:0`, or `--device mps` to select
a device explicitly. Use `--precision float32` or `--precision float64` to fix
precision. Explicit settings are strict: an explicit device never silently moves
to CPU, and explicit float64 never silently becomes float32. For example,
`--device auto --precision float64` may select CPU if the available accelerator
cannot run complex128. CPU double precision remains the reference configuration
for numerical regression tests.

Every CLI run prints a startup summary before reconstruction: data source and
intensity range, wavelength, aperture, object distance, numerical aperture,
effective measurement/output pixel sizes and field of view, reconstruction grid,
geometry, pupil support, solver/epochs/learning rate, requested and resolved
compute settings, Torch version, debugging, and output settings. The summary
uses sampling after preprocessing and the resolved dtypes, rather than just the
requested defaults. `model.requested_compute` retains the request while
`model.compute` holds the resolved device and precision.

Images are transferred to the model's device and precision once at solver entry.
Geometry stays in Torch CPU float64 so integer patch rounding is reproducible;
pupil support is computed there once and transferred before iteration. No global
default dtype/device is changed. Input NumPy arrays remain accepted at API entry,
but results, predictions, and measurements are Torch tensors. Measurement inputs
are copied and detached; treat their owned tensors as read-only, since Torch does
not enforce read-only arrays. `measurements.to(device=..., dtype=...)` creates a
new dataset. The `.npz` result format remains unchanged and records compute settings.

## Debugging reconstruction

Show live updates, sampled every 25 frame updates and at every epoch boundary:

```sh
uv run fp Data/LED_sim/nonspeckle_incoherent_capture --epochs 3 \
    --debug --debug-every 25
```

The default panels show object amplitude, log spectrum magnitude, pupil amplitude,
pupil phase, signed intensity residual, and amplitude RMSE. The current projection
solvers keep the pupil fixed: its display will remain constant until a solver
estimates it. Choose panels explicitly, including measurement, prediction,
object phase, or the current patch correction:

```sh
uv run fp Data/LED_sim/nonspeckle_incoherent_capture --epochs 3 \
    --debug --debug-every 25 \
    --debug-views spectrum pupil_amplitude pupil_phase residual update convergence
```

Save snapshots without opening any GUI, or combine this with `--debug` for both:

```sh
uv run fp Data/LED_sim/nonspeckle_incoherent_capture --epochs 1 \
    --debug-dir debug_frames --debug-every 50
```

No state snapshots are allocated when debugging is disabled. Snapshots are detached
clones on the compute device, created only at the selected interval plus initial
and epoch boundaries. Only rendering transfers them to CPU. Keeping every snapshot
in a custom observer consumes memory, so set a suitable interval for large images.
Closing a live debug window stops its rendering while reconstruction continues.

The Python observer API is independent of plotting and of solver update strategy:

```python
from fp.plotting import ReconstructionVisualizer

visualizer = ReconstructionVisualizer(
    views=("spectrum", "pupil_amplitude", "residual"), show=True,
)
result = SaviReconstructor(model, SolverOptions(epochs=3)).reconstruct(
    measurements, observer=visualizer, observer_interval=25,
)
visualizer.finish()
```

A custom `observer(state)` receives `IterationState`: epoch, update count, centered
spectrum, pupil, optional frame index, optional epoch metrics, and an `extras`
dictionary of tensors. Projection extras include measurement, current prediction,
signed residual, and patch correction. A batch or gradient method can omit the
frame index and supply gradients, pupil corrections, or noise estimates instead.
The separate `callback(metrics)` API still reports completed epochs.

## Adding reconstruction approaches

The `Reconstructor` protocol specifies inputs, output, progress, and observer hooks;
it does not impose a projection update loop or inheritance hierarchy. A Wirtinger
solver should own its parameters, loss, step schedule, noise model, and iteration
strategy, and return the common `ReconstructionResult`.

The shared model now exposes the required Torch building blocks:

- `forward_fields(spectrum, pupil=None)`: differentiable linear field operator A.
- `adjoint_fields(fields, pupil=None)`: Hermitian adjoint A*, with FFT scaling and
  overlap sums. It is not an inverse and does not average over measurements.
- `intensity_loss(spectrum, intensities, pupil=None)`: differentiable half mean
  squared intensity error, without detaching or converting its tensor result.
- `amplitude_rmse(spectrum, intensities, pupil=None)`: evaluate the current pupil
  estimate when reporting reconstruction progress.
- `predict_intensities(object_field, pupil=None)`: differentiable object-field
  prediction, including gradients with respect to a supplied pupil estimate.

Use `torch.autograd.grad(loss, spectrum)` or the explicit adjoint for complex
optimization. `pupil` arguments let the solver own a changing pupil separately
from fixed `pupil_support`. Projection solvers alone run under `torch.no_grad()`;
the model is fully differentiable. `fp.diagnostics.emit_state` creates graph-free
observer snapshots for any future optimizer. Tests check the complex inner-product
identity, the explicit Wirtinger intensity gradient against autograd, and numerical
gradients with respect to both the object spectrum and the pupil.

The future Wirtinger optimizer itself is not implemented in this increment.
Additional acquisition adapters and algorithm changes remain separate work.
