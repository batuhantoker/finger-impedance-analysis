"""Fast, headless demonstration of the EMG analysis pipeline.

The example generates class-dependent flexor and extensor HD-sEMG, extracts
time- and frequency-domain features, computes antagonist activation metrics,
and evaluates movement classification. The stiffness proxy is based on
normalized muscle activation; it is not a physical stiffness measurement.

Run with ``python examples/quickstart.py``.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from finger_impedance import (
    class_map,
    co_contraction_index,
    data_preprocess,
    feature_extraction,
    stiffness_proxy,
)

FS = 1024
EPOCH = 256
N_CHANNELS = 64
N_CLASSES = 5
EPOCHS_PER_CLASS = 6
LOWCUT = 15.0
HIGHCUT = 350.0
RANDOM_STATE = 42


def main() -> None:
    """Run the synthetic end-to-end example."""
    rng = np.random.default_rng(RANDOM_STATE)
    samples_per_class = EPOCHS_PER_CLASS * EPOCH
    labels_raw = np.repeat(np.arange(1, N_CLASSES + 1), samples_per_class)
    n_samples = labels_raw.size

    raw_flexor = np.empty((n_samples, N_CHANNELS), dtype=np.float32)
    raw_extensor = np.empty_like(raw_flexor)
    channel_positions = np.arange(N_CHANNELS)
    centers = np.linspace(6, N_CHANNELS - 7, N_CLASSES)
    flexor_levels = np.array([1.00, 0.85, 0.65, 0.50, 0.35])
    extensor_levels = np.array([0.25, 0.40, 0.70, 0.90, 1.10])

    for class_index, class_label in enumerate(range(1, N_CLASSES + 1)):
        mask = labels_raw == class_label
        class_samples = int(mask.sum())
        time = np.arange(class_samples) / FS

        flexor_profile = 0.25 + np.exp(
            -0.5 * ((channel_positions - centers[class_index]) / 6.0) ** 2
        )
        extensor_profile = 0.25 + np.exp(
            -0.5 * ((channel_positions - centers[::-1][class_index]) / 6.0) ** 2
        )
        flexor_phase = rng.uniform(0, 2 * np.pi, N_CHANNELS)
        extensor_phase = rng.uniform(0, 2 * np.pi, N_CHANNELS)
        flexor_tone = np.sin(2 * np.pi * (45 + 18 * class_index) * time[:, None] + flexor_phase)
        extensor_tone = np.sin(
            2 * np.pi * (60 + 15 * (N_CLASSES - class_index)) * time[:, None] + extensor_phase
        )

        raw_flexor[mask] = (
            flexor_levels[class_index]
            * flexor_profile
            * (0.75 * rng.standard_normal((class_samples, N_CHANNELS)) + 0.25 * flexor_tone)
        )
        raw_extensor[mask] = (
            extensor_levels[class_index]
            * extensor_profile
            * (0.75 * rng.standard_normal((class_samples, N_CHANNELS)) + 0.25 * extensor_tone)
        )

    print(f"Flexor EMG shape:  {raw_flexor.shape}")
    print(f"Extensor EMG shape: {raw_extensor.shape}")

    print("\n--- Preprocessing ---")
    flexor_processed = data_preprocess(raw_flexor, FS, LOWCUT, HIGHCUT)
    extensor_processed = data_preprocess(raw_extensor, FS, LOWCUT, HIGHCUT)

    print("\n--- Feature Extraction ---")
    flexor_features = feature_extraction(flexor_processed, EPOCH, FS)
    extensor_features = feature_extraction(extensor_processed, EPOCH, FS)
    n_segments = flexor_features[0].shape[0]
    feature_matrix = np.concatenate((*flexor_features, *extensor_features), axis=1)
    print(f"Segments extracted: {n_segments}")
    print(f"Feature matrix:     {feature_matrix.shape}")

    epoch_labels = class_map(labels_raw.astype(float), EPOCH)[:n_segments]
    valid_epochs = np.isfinite(epoch_labels)
    epoch_labels = epoch_labels[valid_epochs].astype(int)
    feature_matrix = feature_matrix[valid_epochs]

    flexor_activation = flexor_features[0][:n_segments].mean(axis=1)[valid_epochs]
    extensor_activation = extensor_features[0][:n_segments].mean(axis=1)[valid_epochs]
    reference_activation = max(flexor_activation.max(), extensor_activation.max())
    normalized_flexor = flexor_activation / reference_activation
    normalized_extensor = extensor_activation / reference_activation
    co_contraction = co_contraction_index(normalized_flexor, normalized_extensor)
    activation_stiffness = stiffness_proxy(normalized_flexor, normalized_extensor)

    print("\n--- Antagonist Activation Metrics ---")
    print(
        "Co-contraction index: "
        f"mean={co_contraction.mean():.3f}, range={np.ptp(co_contraction):.3f}"
    )
    print(
        "Stiffness proxy:      "
        f"mean={activation_stiffness.mean():.3f}, range={np.ptp(activation_stiffness):.3f}"
    )

    print("\n--- Classification (Logistic Regression, 5-fold CV) ---")
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
            ),
        ]
    )
    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_val_score(
        model,
        feature_matrix,
        epoch_labels,
        cv=cross_validator,
        scoring="accuracy",
    )
    print(f"Accuracy per fold: {scores.round(3)}")
    print(f"Mean +/- std:      {scores.mean():.3f} +/- {scores.std():.3f}")
    print("\nQuickstart complete.")


if __name__ == "__main__":
    main()
