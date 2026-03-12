# Finger Impedance Analysis Framework

[![CI](https://github.com/sikyabgu/finger-impedance-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/sikyabgu/finger-impedance-analysis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

A computational framework for analyzing human finger impedance modulation, stiffness estimation, and force capabilities using high-density surface EMG (HD-sEMG) and force sensor measurements. Developed as part of an MS thesis in Mechatronics Engineering at Sabanci University.

---

## Overview

This framework processes multi-channel EMG and force data to:

- **Estimate finger stiffness** via transfer function fitting in the frequency domain
- **Extract EMG features** from high-density electrode arrays (8×8 grids for flexor and extensor muscles)
- **Classify finger movements** using machine learning on extracted feature sets
- **Analyze impedance modulation** patterns across different finger movements and subjects

The pipeline supports both the [Hyser](https://physionet.org/content/hd-semg/) public HD-sEMG dataset and custom experimental data collected with the Malesevic protocol.

---

## Installation

### pip (recommended)

```bash
# Clone the repository
git clone https://github.com/sikyabgu/finger-impedance-analysis.git
cd finger-impedance-analysis

# Install the package with core dependencies
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"
```

Optional dependency groups:

| Extra | Installs | Required for |
|-------|----------|--------------|
| `image` | opencv-python, scikit-image | Image-based EMG features |
| `hyser` | samplerate | Hyser dataset resampling |
| `plot` | scienceplots | Publication-quality figures |
| `dev` | pytest, ruff | Development and testing |

### conda

```bash
conda env create -f environment.yml
conda activate finger-impedance
pip install -e .
```

### Docker

```bash
# Build the image
docker build -t finger-impedance-analysis .

# Run an interactive Python shell with the package available
docker run -it --rm finger-impedance-analysis

# Mount local data and run a script
docker run -it --rm -v "$(pwd)/data:/app/data" finger-impedance-analysis \
    python scripts/hyser.py
```

---

## Quick Start

Run the self-contained quickstart script (no data files required):

```bash
python examples/quickstart.py
```

This demonstrates the full pipeline — preprocessing, feature extraction, stiffness estimation, and classification — on synthetic data.

---

## Module Architecture

```
finger_impedance/
├── core/                  # Core algorithms
│   ├── functions.py       # Filtering, feature extraction, stiffness, metrics
│   └── tfestimate.py      # Transfer function estimation (H1/H2/FFT)
│
├── signal/                # Signal processing
│   └── signal_features.py # Activation maps, Mean Shift features
│
├── analysis/              # Analysis pipelines
│   ├── stiffness.py       # Cross-subject stiffness analysis
│   ├── hyser_stiffness.py # Hyser dataset stiffness
│   └── force.py           # Force analysis and averaging
│
├── classification/        # Machine learning
│   ├── emg_classifier.py  # Time/freq domain feature classifier
│   └── image_classifier.py# Image-based feature classifier
│
└── visualization/         # Plotting utilities
    ├── interactive_plot.py # Interactive plots
    ├── figure_plots.py     # Publication figures
    └── hyser_plots.py      # Hyser-specific plots

scripts/
├── main.py                # Main pipeline (Malesevic protocol)
├── malesevic_all.py       # Batch processing — Malesevic subjects
├── hyser.py               # Hyser data loading and feature extraction
├── hyser_all_1dof.py      # Batch processing — Hyser 1-DOF dataset
└── download_hyser.py      # Automated PhysioNet dataset download

examples/
└── quickstart.py          # End-to-end demo on synthetic data

extras/
└── matt.py                # FEA visualization (prosthesis design; unrelated to EMG)
```

---

## Methods

### Stiffness Estimation

Finger stiffness is estimated by fitting transfer functions to force signals in the frequency domain. The `tfestimate` module implements H1, H2, and FFT-based transfer function estimation. A spring model (`K/s`) is fit to the magnitude response at DC to extract the stiffness parameter.

### Signal Processing

Raw EMG signals are preprocessed with:
- Zero-lag Butterworth bandpass filtering (15–350 Hz, 4th order)
- Full-wave rectification
- Configurable sampling rate (default 2048 Hz)

### Feature Extraction

Ten features are extracted per epoch from each EMG channel:

| Feature | Description |
|---------|-------------|
| RMS | Root Mean Square |
| MAV | Mean Absolute Value |
| IAV | Integrated Absolute Value |
| VAR | Variance |
| WL | Waveform Length |
| MF | Mean Frequency |
| PF | Peak Frequency |
| MP | Mean Power |
| TP | Total Power |
| SM | Spectral Moment |

### Classification

Movement classification uses k-fold cross-validation with multiple classifiers including Logistic Regression, SVM, Decision Tree, KNN, LDA, Gaussian Naive Bayes, MLP, Random Forest, and AdaBoost.

### Regression Metrics

Force and stiffness regression is evaluated with R², MAE, MSE, RMSE, nRMSE, and VAF.

---

## Datasets

### Hyser (Public)

The [Hyser dataset](https://physionet.org/content/hd-semg/1.0.0/) provides HD-sEMG recordings from 20 subjects across 34 hand gestures. Download it with:

```bash
export PHYSIONET_USER=your_username
export PHYSIONET_PASSWORD=your_password
python scripts/download_hyser.py --dest data/hyser
```

Then run:
```bash
python scripts/hyser.py
python scripts/hyser_all_1dof.py
```

### Malesevic Protocol (Custom)

Place `.mat` files in a `males/` directory. Each file must contain:

| Field | Shape | Description |
|-------|-------|-------------|
| `emg_extensors` | (samples, rows, cols) | Extensor HD-sEMG grid |
| `emg_flexors` | (samples, rows, cols) | Flexor HD-sEMG grid |
| `force` | (samples, channels) | Force measurements |
| `class` | (samples,) | Movement class labels |

Then run:
```bash
python scripts/main.py
```

---

## Development

```bash
# Install with dev extras
make install-dev

# Run tests
make test

# Lint
make lint

# Build Docker image
make docker-build
```

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{yabgu2025finger,
  author  = {Yabgu, Sik},
  title   = {Finger Impedance Modulation Analysis Using High-Density
             Surface Electromyography},
  school  = {Sabanci University},
  year    = {2025},
  address = {Istanbul, Turkey},
  type    = {M.Sc. Thesis},
}
```

---

## License

MIT License — see [LICENSE.md](LICENSE.md).
