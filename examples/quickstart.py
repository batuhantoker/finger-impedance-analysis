"""Quickstart — full pipeline demonstration using synthetic data.

This script illustrates the complete finger-impedance-analysis workflow:

    1. Generate synthetic HD-sEMG data (8×8 flexor + 8×8 extensor channels)
    2. Preprocess (bandpass filter + rectify)
    3. Extract 10 time/freq domain features per epoch
    4. Estimate epoch-wise stiffness from force signals
    5. Run k-fold classification on extracted features

No real data files are required — everything is synthetic, so you can run this
script immediately after installing the package.

Usage
-----
    python examples/quickstart.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from finger_impedance import (
    class_map,
    data_preprocess,
    feature_extraction,
    force_mean,
)

# ── Configuration ─────────────────────────────────────────────────────────────

FS = 2048          # Sampling frequency [Hz]
DURATION = 10      # Signal duration [s]
N_CHANNELS = 64    # 8×8 HD-sEMG grid (flexors or extensors)
EPOCH = 256        # Samples per analysis window (~125 ms)
N_CLASSES = 5      # Simulated finger movements
N_FORCE_CH = 2     # Force sensor channels
LOWCUT = 15.0      # Bandpass lower cutoff [Hz]
HIGHCUT = 350.0    # Bandpass upper cutoff [Hz]
LOWPASS_CUTOFF = 5.0  # (API compat, not used internally)

rng = np.random.default_rng(42)

# ── Step 1: Synthesise raw signals ────────────────────────────────────────────

n_samples = FS * DURATION

# Raw EMG: Gaussian noise with class-dependent amplitude modulation
raw_emg = rng.standard_normal((n_samples, N_CHANNELS)).astype(np.float32)

# Class labels: repeat each class for equal duration
labels_raw = np.repeat(np.arange(1, N_CLASSES + 1), n_samples // N_CLASSES)
labels_raw = np.pad(labels_raw, (0, n_samples - len(labels_raw)), mode="edge")

# Force signal: sinusoidal + noise, class-dependent amplitude
force_signal = np.zeros((n_samples, N_FORCE_CH))
for c in range(1, N_CLASSES + 1):
    mask = labels_raw == c
    amplitude = 1.0 + 0.5 * c
    t = np.arange(mask.sum()) / FS
    for ch in range(N_FORCE_CH):
        force_signal[mask, ch] = amplitude * np.sin(2 * np.pi * 2 * t) + 0.1 * rng.standard_normal(mask.sum())

print(f"Raw EMG shape:    {raw_emg.shape}")
print(f"Labels shape:     {labels_raw.shape}")
print(f"Force shape:      {force_signal.shape}")

# ── Step 2: Preprocess EMG ────────────────────────────────────────────────────

print("\n--- Preprocessing ---")
emg_processed = data_preprocess(raw_emg, FS, LOWCUT, HIGHCUT, LOWPASS_CUTOFF)
print(f"Processed EMG shape: {emg_processed.shape}")

# ── Step 3: Extract features ──────────────────────────────────────────────────

print("\n--- Feature Extraction ---")
RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM = feature_extraction(emg_processed, EPOCH)
n_segments = RMS.shape[0]
print(f"Segments extracted: {n_segments}")
print(f"Feature shape (each): {RMS.shape}  [segments × channels]")

# Concatenate all 10 features → flat feature matrix
feature_matrix = np.concatenate([RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM], axis=1)
print(f"Full feature matrix: {feature_matrix.shape}  [segments × (10 × channels)]")

# ── Step 4: Epoch-wise labels and force means ─────────────────────────────────

epoch_labels = class_map(labels_raw.astype(float), EPOCH).round().astype(int)
epoch_labels = epoch_labels[:n_segments]

force_means = force_mean(force_signal, EPOCH)
print(f"\nEpoch labels shape:  {epoch_labels.shape}")
print(f"Force means shape:   {force_means.shape}")

# ── Step 5: Classify movements ────────────────────────────────────────────────

print("\n--- Classification (Logistic Regression, 5-fold CV) ---")
scaler = StandardScaler()
X = scaler.fit_transform(feature_matrix[:n_segments])
y = epoch_labels

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(max_iter=500), X, y, cv=cv, scoring="accuracy")
print(f"Accuracy per fold: {scores.round(3)}")
print(f"Mean ± std:        {scores.mean():.3f} ± {scores.std():.3f}")

print("\nQuickstart complete.")
