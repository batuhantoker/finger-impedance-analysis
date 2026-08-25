# Finger Impedance Analysis Framework

[![CI](https://github.com/batuhantoker/finger-impedance-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/batuhantoker/finger-impedance-analysis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

A computational framework for HD-sEMG feature extraction, isometric force analysis, antagonist-muscle activation metrics, movement classification, and stiffness estimation when synchronized displacement and force measurements are available. It was developed as part of an MS thesis in Mechatronics Engineering at Sabanci University.

## Measurement Scope

The included Hyser and Malesevic processing pipelines operate on isometric HD-sEMG and force recordings. They support:

- Windowed force analysis
- `co_contraction_index`, a dimensionless measure of antagonist activation balance
- `stiffness_proxy`, the dimensionless sum of normalized flexor and extensor activation
- EMG feature extraction and movement classification

`stiffness_proxy` is an activation-based proxy, not a stiffness measurement and not a value with physical stiffness units. The Hyser records used here do not contain the synchronized displacement signal needed to identify physical stiffness.

`estimate_stiffness(displacement, force, epoch)` is the physical stiffness API. It fits a force-versus-displacement slope independently for each epoch and channel, requires synchronized arrays with identical shapes, and rejects epochs without measurable displacement variation. With force in newtons and displacement in metres, its result is in N/m.

### Version 0.2 Changes

- Removed the force-only `force_stiffness` calculation because force at a fixed position does not identify stiffness.
- Added `co_contraction_index`, `stiffness_proxy`, and displacement-based `estimate_stiffness`.
- Added sampling frequency as a required `feature_extraction` argument so spectral features are reported in Hz.
- Replaced generated pickle files with validated schema-v2 NPZ archives. Existing pickle outputs must be regenerated from their source data.

## Installation

Python 3.10, 3.11, and 3.12 are supported.

### pip

```bash
git clone https://github.com/batuhantoker/finger-impedance-analysis.git
cd finger-impedance-analysis

# Core library
python -m pip install -e .

# All optional runtime features
python -m pip install -e ".[all]"

# Development tools
python -m pip install -e ".[dev,all]"
```

For a reproducible development environment, install
[uv](https://docs.astral.sh/uv/) and use the committed lockfile:

```bash
uv sync --frozen --extra dev --extra all
```

Optional dependency groups:

| Extra | Installs | Required for |
|-------|----------|--------------|
| `image` | opencv-python, scikit-image | Image-based EMG features |
| `hyser` | wfdb | Reading Hyser WFDB records |
| `malesevic` | mat73 | Reading Malesevic MATLAB files |
| `dev` | build, pytest, pytest-cov, ruff | Building, testing, and linting |
| `all` | image, hyser, malesevic extras | All optional runtime features |

Install `.[hyser]` for the supported Hyser workflow described below.

### conda

```bash
conda env create -f environment.yml
conda activate finger-impedance
python -m pip install -e .
```

The environment file installs dependencies only. The final command installs this checkout, rather than asking a package index for a project distribution that may not exist there.

### Docker

```bash
docker build -t finger-impedance-analysis .
docker run -it --rm finger-impedance-analysis
```

The image installs the package non-editably with all runtime extras. Development and test dependencies are not installed.

To process data mounted from the host:

```bash
docker run --rm -v "$(pwd)/data:/app/data" finger-impedance-analysis \
    python scripts/hyser.py /app/data/hyser --output-dir /app/data/hyser_processed
```

## Quick Start

Run the self-contained, headless example; no data files are required:

```bash
python examples/quickstart.py
```

It demonstrates EMG preprocessing, feature extraction, `co_contraction_index`, the dimensionless activation-based `stiffness_proxy`, and movement classification on synthetic data. It does not estimate physical stiffness.

For synchronized displacement and force data, use:

```python
from finger_impedance import estimate_stiffness

# Both arrays have shape (samples, channels).
stiffness_n_per_m = estimate_stiffness(displacement_m, force_n, epoch=250)
```

## Hyser Workflow

The [Hyser v1.0.0 dataset](https://physionet.org/content/hd-semg/1.0.0/) is open access. The supported workflow uses its preprocessed EMG and force WFDB records from the 1-DoF protocol.

Install the reader dependency first:

```bash
python -m pip install -e ".[hyser]"
```

### Download WFDB Records

The official project downloader reads the PhysioNet manifest, validates paths, downloads atomically, and verifies every file against the published SHA-256 checksum:

```bash
python scripts/download_hyser.py --dest data/hyser
```

By default, this selects the paired preprocessed EMG and force `.hea`/`.dat` records for all 20 subjects and both sessions, approximately 29.3 GB. A smaller verified subset can be selected by repeating `--subject` or `--session`:

```bash
python scripts/download_hyser.py --dest data/hyser --subject 1 --session 1
```

Use `--dry-run` to inspect the selection. Use `--all` only when every file in the PhysioNet manifest is required; it ignores subject and session filters and is substantially larger than the default 1-DoF selection.

### Process WFDB Records

Run the official processor against either the download root or its `1dof_dataset` directory:

```bash
python scripts/hyser.py data/hyser \
    --output-dir data/hyser_processed \
    --window-duration 0.25
```

The processor pairs EMG and force records by subject, session, finger, and sample; uses each WFDB record's sampling frequency; creates non-overlapping windows; and writes one compressed schema-v2 NPZ archive per subject/session. A requested window must map to the same exact duration at both sampling rates; the default 0.25-second window is 512 EMG samples and 25 force samples. The default output names are `subjectNN_sessionN.npz`.

Hyser is an isometric dataset. These outputs contain force and activation-derived metrics, but no physical stiffness estimate.

### Schema-v2 NPZ

Load an archive without pickle support:

```python
import numpy as np

with np.load("data/hyser_processed/subject01_session1.npz", allow_pickle=False) as data:
    assert data["schema_version"].item() == 2
    features = data["features"]
    force = data["force"]
    groups = data["group_ids"]
```

For `N` matched windows, `C` EMG channels, and `F` force channels, each Hyser schema-v2 archive contains:

| Key | Shape | Meaning |
|-----|-------|---------|
| `schema_version` | scalar | Integer schema identifier, currently `2` |
| `label_space` | scalar | Label convention, `hyser_1dof_finger_v1` |
| `subject_id` | scalar | Numeric subject identifier |
| `session_id` | scalar | Numeric session identifier |
| `window_duration_seconds` | scalar | Realized non-overlapping window duration |
| `emg_sampling_frequency` | scalar | Source EMG rate in Hz |
| `force_sampling_frequency` | scalar | Source force rate in Hz |
| `emg_epoch_samples` | scalar | Samples per EMG window |
| `force_epoch_samples` | scalar | Samples per force window |
| `features` | `(N, 10*C)` | RMS, MAV, IAV, variance, WL, MF, PF, MP, TP, and SM blocks |
| `feature_names` | `(10*C,)` | `feature:channel` name for every feature column |
| `labels` | `(N,)` | Finger labels `1` through `5` |
| `force` | `(N, F)` | Mean WFDB force value in each matched window |
| `force_names` | `(F,)` | Source WFDB force-channel names |
| `force_units` | `(F,)` | Units declared by the source WFDB force channels |
| `source_record` | `(N,)` | Source preprocessed EMG record name |
| `group_ids` | `(N,)` | Integer record grouping for leakage-safe data splits |
| `extensor_activation` | `(N,)` | Mean extensor RMS divided by its session reference |
| `flexor_activation` | `(N,)` | Mean flexor RMS divided by its session reference |
| `co_contraction_index` | `(N,)` | Dimensionless antagonist balance in `[0, 1]` |
| `stiffness_proxy` | `(N,)` | Dimensionless sum of normalized flexor and extensor activation |
| `extensor_activation_reference` | scalar | Raw extensor RMS 95th percentile used for normalization |
| `flexor_activation_reference` | scalar | Raw flexor RMS 95th percentile used for normalization |

The `force` units are those declared by the source WFDB force record. The activation values and `stiffness_proxy` are dimensionless; neither should be labelled as N/m or interpreted as directly measured physical stiffness.

Create movement-level force and activation summaries from processed archives with:

```bash
python scripts/hyser_all_1dof.py data/hyser_processed/*.npz \
    --output hyser_activation_proxy_summary.csv
```

## Custom Malesevic Data

Place each `s<subject>.mat` file in one input directory. Each file must contain:

| Field | Shape | Description |
|-------|-------|-------------|
| `emg_extensors` | `(samples, rows, cols)` | Extensor HD-sEMG grid |
| `emg_flexors` | `(samples, rows, cols)` | Flexor HD-sEMG grid |
| `force` | `(samples, channels)` | Isometric force measurements |
| `class` | `(samples,)` | Movement class labels |

Install the MATLAB reader with `python -m pip install -e ".[malesevic]"` before running the converter.

Run the converter with explicit input and output directories:

```bash
python scripts/main.py --input-dir males --output-dir malesevic_processed
```

It writes compressed `data_s<subject>.npz` files with `schema_version=2`, per-muscle feature arrays, force, movement IDs, normalization references, `co_contraction_index`, and the dimensionless `stiffness_proxy`. It does not produce physical stiffness because the input schema has no displacement field.

The converter preserves force values and labels their unit as `raw` by default. If a validated sensor calibration requires an affine conversion, pass `--force-scale`, `--force-offset`, and `--force-unit`. Sampling frequency, filter cutoffs, epoch length, and subject selection are also explicit CLI options; run `python scripts/main.py --help` for details.

Summarize those archives without treating the proxy as measured stiffness:

```bash
python scripts/malesevic_all.py malesevic_processed/*.npz \
    --output malesevic_activation_proxy_summary.csv
```

## Methods

### EMG Processing

Raw EMG preprocessing applies a fourth-order, zero-lag Butterworth bandpass filter along the sample axis. It preserves the signed signal so spectral, variance, and waveform-length features remain valid; amplitude features perform their own absolute-value or square operation. Cutoffs and sampling frequency are caller-configurable. Hyser processing consumes the dataset's preprocessed WFDB EMG records.

Ten features are extracted per epoch and EMG channel:

| Feature | Description |
|---------|-------------|
| RMS | Root mean square |
| MAV | Mean absolute value |
| IAV | Integrated absolute value |
| VAR | Variance |
| WL | Waveform length |
| MF | Mean frequency |
| PF | Peak frequency |
| MP | Mean power |
| TP | Total power |
| SM | Spectral moment |

### Activation Metrics

For normalized flexor activation `f` and extensor activation `e`:

```text
co_contraction_index = 2 * min(f, e) / (f + e)
stiffness_proxy      = f + e
```

The co-contraction index is zero when there is no activation or only one antagonist is active, and one when non-zero antagonist activations are balanced. The proxy tracks total normalized antagonist activation. Both are dimensionless and depend on the chosen activation references.

These are explicit operational metrics, not validated substitutes for mechanical stiffness. Any use as a stiffness surrogate should be calibrated against perturbation-based stiffness measurements for the target population and apparatus.

### Physical Stiffness

`estimate_stiffness` centers synchronized displacement and force within each epoch and computes their least-squares slope. The output unit is force divided by displacement. Meaningful use requires calibrated sensors, temporal synchronization, matching channels, and measurable displacement excitation in every epoch. Set `min_displacement_range` to the calibrated displacement-sensor resolution so unresolved motion is rejected.

The measured-stiffness command accepts a safe NPZ containing `displacement`, `force`, `epoch`, `force_unit`, and `displacement_unit`, then writes a schema-v2 NPZ containing the estimate and explicit units:

```bash
python -m finger_impedance.analysis.force measured_input.npz measured_stiffness.npz
```

The `tfestimate` module also provides generic H1, H2, and FFT transfer-function utilities. A physical interpretation depends on calibrated input/output signals and the selected model; a force-only spectrum is not a stiffness measurement.

### Classification and Metrics

The package includes EMG movement-classification helpers and regression metrics including R2, MAE, MSE, RMSE, normalized RMSE, and VAF.

Evaluate schema-v2 feature archives with grouped cross-validation:

```bash
python -m finger_impedance.classification.emg_classifier \
    data/hyser_processed/subject*_session*.npz
```

Files are grouped by `subject_id` by default, so pass at least two subjects. Use `--group-by record` for within-subject Hyser experiments or `--group-by file` for explicitly independent files. Scaling is fitted independently inside each training fold.

The image-classifier CLI accepts schema-v2 NPZ archives containing fixed-size `canny_ext`, `canny_flex`, `Harris_ext`, and `Harris_flex` arrays plus `movement_id`, `label_space`, and an `image_feature_signature` describing the extraction settings. It rejects inconsistent shapes or signatures instead of truncating features.

## Project Layout

```text
finger_impedance/
|-- core/              # Signal processing, activation metrics, stiffness API, transfer functions
|-- signal/            # Activation-map and image-oriented signal features
|-- classification/    # EMG classifiers
|-- analysis/          # Schema-v2 proxy summaries and measured-stiffness processing
`-- visualization/     # Schema-v2 plotting utilities

scripts/
|-- download_hyser.py  # Manifest-driven, checksum-verified PhysioNet downloader
|-- hyser.py           # WFDB-to-schema-v2 NPZ processor
|-- main.py            # Malesevic-to-schema-v2 NPZ processor
|-- hyser_all_1dof.py  # Hyser activation-proxy summary command
`-- malesevic_all.py   # Malesevic activation-proxy summary command

examples/
`-- quickstart.py      # Headless synthetic EMG example
```

## Development

```bash
make install-dev
make lint
make test
make build
make docker-build
```

`make lint` checks package code, scripts, examples, and tests. `make test` enforces the coverage floor over the critical `finger_impedance.core` logic. `make build` creates both source and wheel distributions in `dist/`.

## Citation

If you use this framework in your research, cite this repository:

```bibtex
@software{toker2025finger,
  author = {Toker, Batuhan},
  title  = {Finger Impedance Analysis Framework},
  url    = {https://github.com/batuhantoker/finger-impedance-analysis},
  year   = {2025},
}
```

## License

MIT License. See [LICENSE.md](LICENSE.md).
