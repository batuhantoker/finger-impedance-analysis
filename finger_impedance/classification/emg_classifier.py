"""Evaluate EMG feature classifiers from safe, versioned NPZ datasets."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SCHEMA_VERSION = 2
SCHEMA_VERSION_KEY = "schema_version"
LABEL_KEYS = ("movement_id", "labels")
METADATA_KEYS = (
    "label_space",
    "window_duration_seconds",
    "emg_sampling_frequency",
    "emg_epoch_samples",
)
MUSCLE_FEATURE_KEYS = (
    "tp_flex",
    "tp_ext",
    "sm_flex",
    "sm_ext",
    "rms_flex",
    "rms_ext",
    "wl_flex",
    "wl_ext",
)


def _validate_schema_version(value: np.ndarray, path: Path) -> None:
    version = np.asarray(value)
    if version.size != 1 or str(version.item()).lower().removeprefix("v") not in {"2", "2.0"}:
        raise ValueError(f"{path}: {SCHEMA_VERSION_KEY!r} must be {SCHEMA_VERSION}")


def _load_feature_file(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, tuple[object, ...]]:
    with np.load(path, allow_pickle=False) as data:
        if SCHEMA_VERSION_KEY not in data:
            raise ValueError(f"{path}: missing required key: {SCHEMA_VERSION_KEY}")
        _validate_schema_version(data[SCHEMA_VERSION_KEY], path)
        label_key = next((key for key in LABEL_KEYS if key in data), None)
        if label_key is None:
            raise ValueError(f"{path}: missing a label array ('movement_id' or 'labels')")
        labels = np.asarray(data[label_key])
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError(f"{path}: {label_key!r} must be a non-empty 1D array")
        if labels.dtype.kind not in "iuf" or not np.all(np.isfinite(labels)):
            raise ValueError(f"{path}: {label_key!r} must contain finite numeric labels")

        missing_metadata = [key for key in METADATA_KEYS if key not in data]
        if missing_metadata:
            raise ValueError(f"{path}: missing metadata: {', '.join(missing_metadata)}")
        metadata = []
        for key in METADATA_KEYS:
            value = np.asarray(data[key])
            if value.size != 1 or value.dtype.kind not in "iufUS":
                raise ValueError(f"{path}: {key!r} must be a numeric or string scalar")
            item = value.item()
            if value.dtype.kind in "iuf" and not np.isfinite(item):
                raise ValueError(f"{path}: {key!r} must be finite")
            if value.dtype.kind in "US" and not str(item):
                raise ValueError(f"{path}: {key!r} must not be empty")
            metadata.append(item)
        if "subject_id" not in data:
            raise ValueError(f"{path}: missing metadata: subject_id")
        subject_id_array = np.asarray(data["subject_id"])
        if subject_id_array.size != 1 or subject_id_array.dtype.kind not in "iu":
            raise ValueError(f"{path}: 'subject_id' must be an integer scalar")
        subject_id = int(subject_id_array.item())

        if "features" in data:
            features = np.asarray(data["features"])
            if "feature_names" not in data:
                raise ValueError(f"{path}: dense 'features' require 'feature_names'")
            feature_names = np.asarray(data["feature_names"])
            if feature_names.ndim != 1 or feature_names.dtype.kind not in "US":
                raise ValueError(f"{path}: 'feature_names' must be a one-dimensional string array")
            feature_signature: tuple[object, ...] = tuple(feature_names.astype(str))
        else:
            missing = [key for key in MUSCLE_FEATURE_KEYS if key not in data]
            if missing:
                raise ValueError(
                    f"{path}: missing 'features' or muscle feature keys: {', '.join(missing)}"
                )
            arrays = [np.asarray(data[key]) for key in MUSCLE_FEATURE_KEYS]
            if any(array.ndim != 2 for array in arrays):
                raise ValueError(f"{path}: muscle features must have shape (samples, features)")
            processing_keys = (
                "emg_layout",
                "extensor_grid_shape",
                "flexor_grid_shape",
                "filter_lowcut_hz",
                "filter_highcut_hz",
            )
            missing_processing = [key for key in processing_keys if key not in data]
            if missing_processing:
                raise ValueError(
                    f"{path}: missing preprocessing metadata: {', '.join(missing_processing)}"
                )
            layout = np.asarray(data["emg_layout"])
            extensor_shape_array = np.asarray(data["extensor_grid_shape"])
            flexor_shape_array = np.asarray(data["flexor_grid_shape"])
            lowcut = np.asarray(data["filter_lowcut_hz"])
            highcut = np.asarray(data["filter_highcut_hz"])
            if layout.size != 1 or layout.dtype.kind not in "US":
                raise ValueError(f"{path}: 'emg_layout' must be a string scalar")
            if (
                extensor_shape_array.ndim != 1
                or flexor_shape_array.ndim != 1
                or extensor_shape_array.size != 2
                or flexor_shape_array.size != 2
                or np.any(extensor_shape_array <= 0)
                or np.any(flexor_shape_array <= 0)
            ):
                raise ValueError(f"{path}: EMG grid shapes must contain two positive dimensions")
            if lowcut.size != 1 or highcut.size != 1:
                raise ValueError(f"{path}: filter cutoffs must be numeric scalars")
            lowcut_value = float(lowcut.item())
            highcut_value = float(highcut.item())
            emg_fs = float(metadata[2])
            if not 0 < lowcut_value < highcut_value < emg_fs / 2:
                raise ValueError(f"{path}: invalid filter cutoff metadata")
            extensor_shape = tuple(extensor_shape_array.astype(int))
            flexor_shape = tuple(flexor_shape_array.astype(int))
            feature_signature = (
                "muscle",
                str(layout.item()),
                extensor_shape,
                flexor_shape,
                lowcut_value,
                highcut_value,
                *(tuple((key, array.shape[1]) for key, array in zip(MUSCLE_FEATURE_KEYS, arrays))),
            )
            features = np.concatenate(arrays, axis=1)

        if features.ndim != 2 or features.shape[0] != labels.size or features.shape[1] == 0:
            raise ValueError(f"{path}: features must have shape ({labels.size}, n_features)")
        if features.dtype.kind not in "biuf" or not np.all(np.isfinite(features)):
            raise ValueError(f"{path}: features must contain finite numeric values")
        if "features" in data and len(feature_signature) != features.shape[1]:
            raise ValueError(f"{path}: 'feature_names' must contain one name per feature column")

        groups = None
        if "group_ids" in data:
            groups = np.asarray(data["group_ids"])
            if groups.ndim != 1 or groups.size != labels.size:
                raise ValueError(f"{path}: 'group_ids' must contain one value per sample")
            if groups.dtype.kind not in "biuf" or not np.all(np.isfinite(groups)):
                raise ValueError(f"{path}: 'group_ids' must contain finite numeric values")

    signature = (*metadata, *feature_signature)
    return features.astype(float, copy=False), labels, groups, subject_id, signature


def load_feature_files(
    paths: Sequence[str | Path],
    *,
    group_by: str = "subject",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load compatible schema-v2 feature files and assign validation groups."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    input_paths = [Path(path) for path in paths]
    if not input_paths:
        raise ValueError("at least one NPZ file is required")
    if group_by not in {"subject", "file", "record"}:
        raise ValueError("group_by must be 'subject', 'file', or 'record'")

    feature_sets = []
    label_sets = []
    group_sets = []
    expected_signature = None
    next_group_id = 0
    for path in input_paths:
        features, labels, local_groups, subject_id, signature = _load_feature_file(path)
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise ValueError(
                f"{path}: feature metadata signature {signature} does not match "
                f"{expected_signature}"
            )
        feature_sets.append(features)
        label_sets.append(labels)
        if group_by == "subject":
            group_sets.append(np.full(labels.size, subject_id, dtype=int))
        elif group_by == "file" or local_groups is None:
            group_sets.append(np.full(labels.size, next_group_id, dtype=int))
            next_group_id += 1
        else:
            _, remapped = np.unique(local_groups, return_inverse=True)
            group_sets.append(remapped + next_group_id)
            next_group_id += int(remapped.max()) + 1

    return (
        np.concatenate(feature_sets, axis=0),
        np.concatenate(label_sets),
        np.concatenate(group_sets),
    )


def _make_cross_validator(
    labels: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
    random_state: int,
) -> tuple[StratifiedGroupKFold | StratifiedKFold, np.ndarray | None]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    classes, class_counts = np.unique(labels, return_counts=True)
    if classes.size < 2:
        raise ValueError("classification requires at least two classes")

    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or groups.size != labels.size:
            raise ValueError("groups must be a 1D array with one value per sample")
        unique_groups = np.unique(groups)
        if unique_groups.size < 2:
            raise ValueError("grouped CV requires at least two independent groups")
        class_group_counts = [np.unique(groups[labels == label]).size for label in classes]
        folds = min(n_splits, unique_groups.size, min(class_group_counts))
        if folds < 2:
            raise ValueError("each class must occur in at least two groups for grouped CV")
        return (
            StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=random_state,
            ),
            groups,
        )

    folds = min(n_splits, int(class_counts.min()))
    if folds < 2:
        raise ValueError("each class must contain at least two samples for stratified CV")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state), None


def evaluate_models(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 10,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Return fold-wise accuracy scores for deterministic EMG classifiers."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array")
    if labels.ndim != 1 or labels.size != features.shape[0]:
        raise ValueError("labels must be a 1D array with one value per sample")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must contain only finite values")

    cross_validator, cv_groups = _make_cross_validator(labels, groups, n_splits, random_state)
    estimators = (
        (
            "LogReg",
            LogisticRegression(max_iter=1000, random_state=random_state),
        ),
        ("SVM", SVC(random_state=random_state)),
    )

    scores = {}
    for name, estimator in estimators:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ]
        )
        scores[name] = cross_val_score(
            pipeline,
            features,
            labels,
            groups=cv_groups,
            cv=cross_validator,
            scoring="accuracy",
            error_score="raise",
        )
    return scores


def _save_score_plot(
    scores: dict[str, np.ndarray],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    import matplotlib

    if not output_path.suffix:
        raise ValueError("plot output path must include a file extension")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(list(scores.values()))
    axis.set_xticks(range(1, len(scores) + 1), list(scores))
    axis.set_title("EMG Classifier Comparison")
    axis.set_ylabel("Accuracy (0-1)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate EMG classifiers from schema-v2 NPZ feature files."
    )
    parser.add_argument("files", nargs="+", type=Path, help="input schema-v2 NPZ files")
    parser.add_argument("--folds", type=int, default=10, help="maximum CV fold count")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--group-by",
        choices=("subject", "file", "record"),
        default="subject",
        help="independent unit kept together during cross-validation",
    )
    parser.add_argument("--plot", type=Path, help="optional path for an accuracy boxplot")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing plot")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line classifier evaluation."""
    args = _build_parser().parse_args(argv)
    features, labels, groups = load_feature_files(args.files, group_by=args.group_by)
    scores = evaluate_models(
        features,
        labels,
        groups=groups,
        n_splits=args.folds,
        random_state=args.random_state,
    )

    print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
    for name, model_scores in scores.items():
        print(f"{name}: accuracy={model_scores.mean():.3f} +/- {model_scores.std():.3f}")
    if args.plot:
        _save_score_plot(scores, args.plot, overwrite=args.overwrite)
        print(f"Saved plot to {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
