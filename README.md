# Finger Impedance Analysis Framework

A computational framework for analyzing human finger impedance modulation, stiffness estimation, and force capabilities using high-density surface EMG (HD-sEMG) and force sensor measurements. Developed as part of an MS thesis in Mechatronics Engineering at Sabanci University.

## Overview

This framework processes multi-channel EMG and force data to:

- **Estimate finger stiffness** via transfer function fitting in the frequency domain
- **Extract EMG features** from high-density electrode arrays (8×8 grids for flexor and extensor muscles)
- **Classify finger movements** using machine learning on extracted feature sets
- **Analyze impedance modulation** patterns across different finger movements and subjects

The pipeline supports both the [Hyser](https://physionet.org/content/hd-semg/) public HD-sEMG dataset and custom experimental data collected with the Malesevic protocol.

## Methods

### Stiffness Estimation

Finger stiffness is estimated by fitting transfer functions to force signals in the frequency domain. The `tfestimate` module implements H1, H2, and FFT-based transfer function estimation methods. A spring model (`K/s`) is fit to the magnitude response at DC to extract the stiffness parameter, with optional spring-damper and mass-spring-damper models for higher-order fits.

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

Movement classification is evaluated using k-fold cross-validation with multiple classifiers including Logistic Regression, SVM, Decision Tree, KNN, LDA, Gaussian Naive Bayes, MLP, Random Forest, and AdaBoost.

### Regression Metrics

Force and stiffness regression is evaluated with R², MAE, MSE, RMSE, nRMSE, and VAF metrics.

## Modules

| File | Description |
|------|-------------|
| `main.py` | Main pipeline — loads subject data (Malesevic format), preprocesses EMG, extracts features, estimates stiffness, and saves per-subject pickle files |
| `functions.py` | Core library — filtering, feature extraction, stiffness estimation, classification, and regression metrics |
| `tfestimate.py` | Transfer function estimation class with H1/H2/FFT methods, Bode plot generation, and pole-zero optimization |
| `hyser.py` | Data loading and feature extraction for the Hyser HD-sEMG dataset |
| `hyser_stiffness.py` | Stiffness visualization and cross-subject analysis for Hyser data |
| `hyser_all_1dof.py` | Batch processing for Hyser 1-DOF dataset across subjects |
| `signal_features.py` | Standalone EMG feature extraction with Mean Shift clustering features |
| `classifier.py` | Movement classification pipeline with multiple ML models (image-based features) |
| `classifier2.py` | Movement classification using time/frequency domain EMG features |
| `stiffness.py` | Stiffness analysis and visualization across subjects with Bode plot fitting |
| `force.py` | Cross-subject force and stiffness averaging with statistical visualization |
| `figure_plots.py` | Publication figure generation |
| `interactive_plot.py` | Interactive plotting utilities |
| `matt.py` | Additional analysis scripts |

## Usage

### Processing Subject Data (Malesevic Protocol)

```python
# main.py processes subjects 1-20, extracting features and stiffness
# Expects .mat files in males/ directory
python main.py
```

This produces per-subject pickle files (`data_s{N}.pkl`) containing EMG features, force measurements, movement labels, and estimated stiffness values.

### Processing Hyser Dataset

```python
# Download Hyser data from PhysioNet first
# https://physionet.org/content/hd-semg/
python hyser.py
```

### Running Classification

```python
# Requires preprocessed pickle files from main.py or hyser.py
python classifier2.py
```

## Data Format

Input `.mat` files are expected to contain:
- `emg_extensors` — 3D array (samples × rows × cols) of extensor EMG
- `emg_flexors` — 3D array of flexor EMG
- `force` — 2D array (samples × channels) of force measurements
- `class` — Movement class labels

## Dependencies

```
numpy
scipy
scikit-learn
matplotlib
mat73
h5py
pickle
```

Optional: `samplerate`, `scienceplots`, `opencv-python` (for image-based features)

## License

MIT License — see [LICENSE.md](LICENSE.md).
