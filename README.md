# fp

Torch-based Fourier ptychography with calibrated acquisition geometry, joint object/pupil
projection solvers, and fixed-pupil Wirtinger flow.

## Install and run

Requires Python 3.14 or newer.

```sh
uv sync
uv run fp Data/LED_sim/nonspeckle_incoherent_capture --epochs 10 --output result.npz
```

`python -m fp` provides the same command. Dataset `config.toml` values are defaults;
for a `.mat` input, the default is a sibling TOML file with the same stem.
`--config PATH` selects a settings file explicitly. Command-line arguments override
TOML values. Use `uv run fp --help` for all options.
Startup output reports optics, sampling, geometry, solver controls, pupil support,
resolved device and dtypes, and debug/output settings.

## Input and configuration

| Source | Input / geometry |
| --- | --- |
| `array_sim` | Numbered intensity files with spiral array geometry |
| `sim` | Synthetic measurements generated from an amplitude image |
| `matlab` | MATLAB `imlow_HDR` intensity cube and LED calibration |
| `led` | Numbered intensity files with LED calibration |

Reconstruction uses one wavelength and a circular nominal aperture. Specify its
physical `aperture_diameter` or an `objective_na`; the latter determines the
equivalent aperture for the configured object distance.

Array files are named `0000.npy`, `0001.npy`, etc., in spiral acquisition order.
For numbered files, `--frame-pattern '{index}'` selects unpadded names, and
`--frame-pattern 'ILED_{index:02d}'` selects stems such as `ILED_00`. The extension
is supplied separately.
TIFF and other grayscale raster formats can be selected with `--file_extension`.
NPY values retain their intensity scale. Integer rasters are divided by their
dtype's maximum, including 16-bit TIFFs. Images are center-cropped to a square and
resized to the requested resolution, preserving the cropped physical field of view.

A flat dataset `config.toml` can contain:

```toml
source = "array_sim"
file_extension = ".npy"
samples_count = 225
resolution = 300
upsample = 2
wavelength = 532.0
pixel_size_um = 1.85
aperture_diameter = 79.99
translation = 28
object_distance = 2.0
algorithm = "savi"
epochs = 10
learning = 1.0
pupil_learning = 1.0
```

| Setting | Meaning / units |
| --- | --- |
| `samples_count` | Perfect-square number of array/simulation frames; LED inputs use the calibration count |
| `resolution` | Measurement width and height after preprocessing, pixels |
| `upsample` | Integer ratio of output dimensions to measurement dimensions |
| `wavelength` | Float wavelength, nanometers |
| `pixel_size_um` | Native sample-plane pixel spacing, micrometers |
| `aperture_diameter` | Nominal circular aperture diameter, millimeters |
| `objective_na` | Objective numerical aperture, instead of `aperture_diameter` |
| `translation` | Adjacent array spacing, millimeters |
| `object_distance` | Object-to-aperture distance, meters |
| `learning` / `pupil_learning` | Object / pupil update relaxation, positive; default 1 |
| `object_epsilon` / `pupil_epsilon` | Positive denominator stabilizers; default `1e-10` |

For `--source sim`, `--data_name` selects an amplitude image without its extension.
The image is converted to grayscale and resized to the object grid;
`pixel_size_um` then specifies the simulated measurement spacing.

### MATLAB/HDR and calibrated LED inputs

```sh
uv run fp Data/USAF_red.mat --config configs/usaf_red.toml --epochs 10 --output usaf.npz
uv run fp Data/MouseKidney_green.mat --config configs/mousekidney_green.toml --epochs 10
uv run fp Data/USAF_red --config configs/usaf_red_png.toml --epochs 10
```

Passing a `.mat` input selects `matlab` automatically. A directory can instead use
`--source matlab --data_name STEM`. Cubes have shape `(height, width, frames)`;
`--mat-key` changes the default variable name `imlow_HDR`. Both integer and floating
MATLAB intensities retain their scale, including HDR values above one. Use
`--intensity-scale` for an explicit multiplier. MATLAB v4/v6/v7 through v7.2 files
are supported; v7.3/HDF5 files must be saved in an earlier MATLAB format.

The default companion calibration for `STEM.mat` is `STEM_LED_locations.mat`.
`--led-calibration PATH` selects another file, relative to the dataset directory
(or the input file's parent). A `led` source requires this option. Calibration
files contain either:

- `xlocation_s` and `ylocation_s`: equally sized row or column vectors of LED grid
  indices, in acquisition order. Columns are `(x,y)`; no spiral reordering is added.
- `wavevectors_rad_per_m`: calibrated `(frames,2)` transverse wavevectors `(kx,ky)`.
  These are used directly, with no location transformation or refraction.

Location calibration defaults are 4 mm LED spacing, 90.88 mm LED-to-slide air gap,
1 mm glass thickness, and glass refractive index 1.52. Set these with
`--led-spacing-mm`, `--led-height-mm`, `--glass-thickness-mm`, and
`--glass-refractive-index`. Zero glass thickness disables refraction. Optional
`--led-rotation-deg` rotates grid positions, then `--led-offset-x-mm` and
`--led-offset-y-mm` translate them. TOML uses the corresponding underscore names.
The presets use the UConn acquisition parameters; adjust them to match the rig.

The adapter solves the ray displacement through air and glass using Snell's law.
Positive LED coordinates produce negative incident transverse wavevectors. It
converts direction cosines directly to wavevectors, with no extra sine operation.
Frame counts must match the calibration; preprocessing preserves the cropped field
of view and updates measurement pixel spacing.

## Reconstruction algorithms

Pupil recovery is enabled for the two projection algorithms. The initial pupil is
the nominal circular aperture; the initial object is the Fourier transform of resized mean
measurement intensity. Updates retain the complex object and pupil.

- `--algorithm savi` (default): joint phase retrieval using Eqs. (6)-(11) of
  Holloway et al., [SAVI, Science Advances 3, e1602564 (2017)](https://doi.org/10.1126/sciadv.1602564).
  Each measurement produces one adaptive object correction and one pupil
  correction from the same pre-update fields. The pupil step uses the maximum
  magnitude of the entire object spectrum. The paper's enlarged pupil support
  is implemented as a disk of twice the nominal radius and diameter, clipped to
  the measurement grid. The paper's separate speckle-denoising stage is not applied.
- `--algorithm alternating_projection`: joint object/pupil amplitude projection,
  with corrections normalized by the maximum squared pupil and object-patch
  magnitudes. Pupil support is the nominal aperture.
- `--algorithm wirtinger_flow`: fixed-pupil batch descent on
  `L(z) = mean((abs(Az)**2 - I)**2) / 2`, where `z` is the centered object spectrum.
  It uses the intensity-loss Wirtinger formulation of
  [Candès, Li, and Soltanolkotabi (2015)](https://arxiv.org/abs/1407.1065), adapted
  to the package optical operator. The complex gradient is
  `2 A*((abs(Az)**2 - I) * Az) / I.numel()`, including FFT normalization and overlap
  sums. A curvature-scaled starting step and backtracking enforce intensity-loss
  descent. Each epoch is one full-data update; fields are processed one at a time.
  It uses resized mean-intensity initialization or a supplied `initial_field`.
  A supplied `initial_pupil` is held fixed within the nominal support. Pupil learning
  and projection stabilizer settings do not affect this solver. Amplitude RMSE is
  reported for comparison; the minimized objective is intensity error.

For projection algorithms, `--learning` and `--pupil-learning` control the two update
relaxations. Unit values use the full corrections; for SAVI these are the published Eqs. (10)-(11).
`--object-epsilon` and `--pupil-epsilon` set SAVI's tau1/tau2 stabilizers and
protect the normalized alternating-projection updates from zero denominators.

## Device and precision

Device and precision default to `auto`. Available devices are tried in order:
CUDA, MPS, CPU. Each device is tested with float64/complex128 first, followed by
float32/complex64. This favors an accelerator and its highest supported complex
FFT precision; it does not benchmark individual image sizes.

`--device cpu`, `--device cuda[:index]`, or `--device mps` selects a device.
`--precision float32` or `--precision float64` fixes precision. Explicit requests
are strict; unsupported combinations fail rather than silently changing them.
`--device cpu --precision float64` is the numerical reference configuration.

## Results and debugging

`--output result.npz` saves `object_field`, `pupil`, frame IDs, wavevectors, epoch
numbers, amplitude RMSE, and JSON settings, including acquisition/calibration,
pupil recovery support, and compute settings. Load with `numpy.load(path, allow_pickle=False)`.
`--show` displays the final object amplitude and phase.

`--debug` opens live reconstruction views. `--debug-every 25` samples every 25
measurement updates, plus initialization and epoch boundaries. `--debug-dir PATH`
saves PNG snapshots and works without a GUI; it can be combined with `--debug`.

Default views show object amplitude, log spectrum magnitude, recovered pupil
amplitude/phase, signed intensity residual, and amplitude RMSE. Select panels with
`--debug-views`: `amplitude`, `phase`, `spectrum`, `pupil_amplitude`, `pupil_phase`,
`measurement`, `prediction`, `residual`, `update`, `pupil_update`, `convergence`.
Snapshots are detached copies and cannot change solver state. No snapshots are
allocated when debugging is disabled. Closing the live window lets solving continue.

## Package structure

| Module | Responsibility |
| --- | --- |
| `config.py`, `validation.py`, `tensors.py` | Physical/grid/solver settings and compute resolution |
| `data.py`, `io/`, `preprocessing.py` | Measurement ownership, file decoding, sampling |
| `geometry.py` | Acquisition ordering and transverse wavevectors |
| `fft.py`, `optics.py` | Centered FFTs, pupil supports, differentiable fields/losses/adjoint |
| `reconstructors/` | Projection and Wirtinger algorithms, shared initialization/results, and common protocol |
| `simulation.py` | Synthetic measurements through the optical model |
| `diagnostics.py`, `plotting.py` | State snapshots and live/file visualization |
| `arguments.py`, `cli.py` | CLI/TOML parsing, acquisition setup, and run orchestration |
| `results.py`, `reporting.py` | Numerical export and run summary |

## Development and numerical conventions

```sh
uv run python -B -m unittest discover -s tests -v
uvx ruff check src
uvx ruff format --check src
```

Tests cover the paper update equations, synthetic pupil recovery, support bounds,
state isolation, FFT scaling, adjoints and complex gradients, HDR and raster I/O,
LED refraction, Wirtinger descent, device selection, and CLI reporting.

Validate available local acquisitions at native resolution with all three solvers:

```sh
FP_TEST_DATASETS=1 uv run python -B -m unittest discover -s tests -p test_datasets.py -v
```

These checks use `configs/` presets for USAF HDR, MouseKidney HDR, numbered USAF LED
PNGs, and the nonspeckle array capture. Each verifies 225-frame reconstruction,
finite object/pupil results, decreasing amplitude RMSE over three epochs, and NPZ
export. Missing local datasets are skipped. These are acquisition/solver smoke
checks; they do not establish ground-truth image accuracy or optical calibration.

Measurements are intensity tensors `(frames, height, width)`. Geometry columns are
`(kx, ky)` in rad/m; image axes are `(y, x)`. Internal lengths are meters. Spectra
are centered with Torch's default backward FFT normalization; measured fields are
`ifft2c(object_spectrum_patch * pupil) / upsample**2`. Reconstructed pixel spacing
is the effective measurement spacing divided by `upsample`.

A reconstructor implements `Reconstructor.reconstruct(...)`, accepts a
`MeasurementSet`, and returns `ReconstructionResult`. `initial_field` and
`initial_pupil` accept numerical starting estimates; `callback(metrics)` reports
epochs, and `observer(state)` receives detached spectrum, pupil, and diagnostics.
Updates belong in the reconstructor; optical operators and file decoding remain
separate. Keep custom pupils local to the solver and pass them to model methods.
Metrics must use the current pupil estimate. Never mutate the shared model's
nominal pupil or support. Model operators retain autograd; explicit solver loops use
`torch.no_grad()`. NumPy is used at array/MATLAB file boundaries and for display conversion.
