"""Evaluate image-feature classifiers from fixed-size NPZ arrays."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SCHEMA_VERSION = 2
LABEL_KEY = "movement_id"
FEATURE_KEYS = ("canny_ext", "canny_flex", "Harris_ext", "Harris_flex")
REQUIRED_KEYS = (
    "schema_version",
    "label_space",
    "image_feature_signature",
    LABEL_KEY,
    *FEATURE_KEYS,
)


def flatten_feature_arrays(*feature_arrays: np.ndarray) -> np.ndarray:
    """Flatten and concatenate fixed-size per-sample feature arrays."""
    if not feature_arrays:
        raise ValueError("at least one feature array is required")

    sample_count = np.asarray(feature_arrays[0]).shape[0]
    flattened = []
    for array in feature_arrays:
        array = np.asarray(array)
        if array.ndim < 2 or array.shape[0] != sample_count:
            raise ValueError("feature arrays must have matching sample axes")
        if 0 in array.shape[1:]:
            raise ValueError("feature arrays must have non-empty fixed feature shapes")
        flattened.append(array.reshape(sample_count, -1))
    return np.concatenate(flattened, axis=1)


def _load_image_file(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, tuple[object, ...]]]:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"{path}: missing required keys: {', '.join(missing)}")
        version = np.asarray(data["schema_version"])
        if version.size != 1 or int(version.item()) != SCHEMA_VERSION:
            raise ValueError(f"{path}: 'schema_version' must be {SCHEMA_VERSION}")
        label_space = np.asarray(data["label_space"])
        extraction_signature = np.asarray(data["image_feature_signature"])
        if label_space.size != 1 or label_space.dtype.kind not in "US":
            raise ValueError(f"{path}: 'label_space' must be a string scalar")
        if extraction_signature.size != 1 or extraction_signature.dtype.kind not in "US":
            raise ValueError(f"{path}: 'image_feature_signature' must be a string scalar")

        labels = np.asarray(data[LABEL_KEY])
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError(f"{path}: {LABEL_KEY!r} must be a non-empty 1D array")
        if labels.dtype.kind not in "iuf" or not np.all(np.isfinite(labels)):
            raise ValueError(f"{path}: {LABEL_KEY!r} must contain finite numeric labels")

        arrays = {}
        shapes = {}
        for key in FEATURE_KEYS:
            array = np.asarray(data[key])
            if array.ndim < 2:
                raise ValueError(f"{path}: {key!r} must include a sample and feature axis")
            if array.shape[0] != labels.size:
                raise ValueError(
                    f"{path}: {key!r} has {array.shape[0]} samples; expected {labels.size}"
                )
            if 0 in array.shape[1:]:
                raise ValueError(f"{path}: {key!r} must have a non-empty fixed feature shape")
            if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
                raise ValueError(f"{path}: {key!r} must contain finite numeric values")
            arrays[key] = array.astype(float, copy=False)
            shapes[key] = array.shape[1:]

        groups = None
        if "group_ids" in data:
            groups = np.asarray(data["group_ids"])
            if groups.ndim != 1 or groups.size != labels.size:
                raise ValueError(f"{path}: 'group_ids' must contain one value per sample")
            if groups.dtype.kind not in "biuf" or not np.all(np.isfinite(groups)):
                raise ValueError(f"{path}: 'group_ids' must contain finite numeric values")

    if shapes["canny_ext"] != shapes["canny_flex"]:
        raise ValueError(f"{path}: flexor and extensor Canny arrays must have matching shapes")
    if shapes["Harris_ext"] != shapes["Harris_flex"]:
        raise ValueError(f"{path}: flexor and extensor Harris arrays must have matching shapes")

    features = flatten_feature_arrays(*(arrays[key] for key in FEATURE_KEYS))
    shapes["label_space"] = (str(label_space.item()),)
    shapes["image_feature_signature"] = (str(extraction_signature.item()),)
    return features, labels, groups, shapes


def load_feature_files(
    paths: Sequence[str | Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load fixed-size image arrays and assign one group ID per input file."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    input_paths = [Path(path) for path in paths]
    if not input_paths:
        raise ValueError("at least one NPZ file is required")

    feature_sets = []
    label_sets = []
    group_sets = []
    expected_shapes = None
    next_group_id = 0
    for path in input_paths:
        features, labels, local_groups, shapes = _load_image_file(path)
        if expected_shapes is None:
            expected_shapes = shapes
        elif shapes != expected_shapes:
            raise ValueError(f"{path}: feature schema {shapes} does not match {expected_shapes}")
        feature_sets.append(features)
        label_sets.append(labels)
        if local_groups is None:
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
    """Return fold-wise accuracy scores for deterministic image classifiers."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2D array")
    if labels.ndim != 1 or labels.size != features.shape[0]:
        raise ValueError("labels must be a 1D array with one value per sample")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must contain only finite values")

    cross_validator, cv_groups = _make_cross_validator(labels, groups, n_splits, random_state)
    if cv_groups is None:
        splits = list(cross_validator.split(features, labels))
    else:
        splits = list(cross_validator.split(features, labels, cv_groups))
    minimum_training_samples = min(len(train_indices) for train_indices, _ in splits)
    estimators = (
        (
            "LogReg",
            LogisticRegression(max_iter=1000, random_state=random_state),
        ),
        ("DecTree", DecisionTreeClassifier(random_state=random_state)),
        ("KNN", KNeighborsClassifier(n_neighbors=min(15, minimum_training_samples))),
        ("LinDisc", LinearDiscriminantAnalysis()),
        ("GaussianNB", GaussianNB()),
        (
            "MLPC",
            MLPClassifier(
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=random_state,
            ),
        ),
        ("RFC", RandomForestClassifier(random_state=random_state)),
        ("ABC", AdaBoostClassifier(random_state=random_state)),
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
            cv=splits,
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

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.boxplot(list(scores.values()))
    axis.set_xticks(range(1, len(scores) + 1), list(scores))
    axis.set_title("Image Feature Classifier Comparison")
    axis.set_ylabel("Accuracy (0-1)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate classifiers from fixed-size Canny/Harris NPZ arrays."
    )
    parser.add_argument("files", nargs="+", type=Path, help="input NPZ feature files")
    parser.add_argument("--folds", type=int, default=10, help="maximum CV fold count")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--plot", type=Path, help="optional path for an accuracy boxplot")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing plot")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line classifier evaluation."""
    args = _build_parser().parse_args(argv)
    features, labels, groups = load_feature_files(args.files)
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
