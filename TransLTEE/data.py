"""Dataset loading, validation, splitting, and normalization."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    features: np.ndarray
    treatment: np.ndarray
    outcomes: np.ndarray
    ground_truth_ate: np.ndarray


@dataclass(frozen=True)
class DatasetSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class NormalizationStats:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    outcome_mean: float
    outcome_scale: float

    def transform_features(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.feature_mean) / self.feature_scale).astype(np.float32)

    def transform_outcomes(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.outcome_mean) / self.outcome_scale).astype(np.float32)

    def inverse_outcomes(self, values: np.ndarray) -> np.ndarray:
        return values * self.outcome_scale + self.outcome_mean


def _load_news_features(
    path: Path, rows: int = 5000, columns: int = 3477
) -> np.ndarray:
    """Load NEWS sparse triplets (document, term, count) into a dense matrix."""
    triplets = np.loadtxt(path, delimiter=",", dtype=np.int32)
    if triplets.ndim != 2 or triplets.shape[1] != 3:
        raise ValueError(f"Expected three-column sparse NEWS data in {path}")

    row_indices = triplets[:, 0] - 1
    column_indices = triplets[:, 1] - 1
    if (
        row_indices.min() < 0
        or row_indices.max() >= rows
        or column_indices.min() < 0
        or column_indices.max() >= columns
    ):
        raise ValueError(
            f"Sparse NEWS indices are outside the expected shape in {path}"
        )

    features = np.zeros((rows, columns), dtype=np.float32)
    features[row_indices, column_indices] = triplets[:, 2]
    return features


def load_dataset(
    name: str,
    data_dir: Path,
    repetition: int = 1,
    sequence_length: int = 100,
) -> DatasetBundle:
    """Load one of the ten committed IHDP or NEWS realizations."""
    normalized_name = name.lower()
    if normalized_name == "ihdp":
        raw_path = data_dir / "IHDP" / "csv" / f"ihdp_npci_{repetition}.csv"
        raw = np.loadtxt(raw_path, delimiter=",", dtype=np.float32)
        features = raw[:, 5:]
        dataset_dir = data_dir / "IHDP"
    elif normalized_name == "news":
        dataset_dir = data_dir / "NEWS"
        feature_path = (
            dataset_dir / f"topic_doc_mean_n5000_k3477_seed_{repetition}.csv.x"
        )
        features = _load_news_features(feature_path)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    series_path = dataset_dir / f"Series_y_{repetition}.txt"
    ground_truth_path = dataset_dir / f"Series_groundtruth_{repetition}.txt"
    series = np.loadtxt(series_path, delimiter=",", dtype=np.float32)
    ground_truth = np.loadtxt(ground_truth_path, delimiter=",", dtype=np.float32)

    treatment = series[:, 0]
    outcomes = series[:, 1 : sequence_length + 1]
    ground_truth = ground_truth[:sequence_length]

    if features.shape[0] != outcomes.shape[0]:
        raise ValueError(
            f"Feature/outcome row mismatch: {features.shape[0]} != {outcomes.shape[0]}"
        )
    if outcomes.shape[1] != sequence_length or ground_truth.shape[0] != sequence_length:
        raise ValueError(
            f"Requested {sequence_length} timesteps, but committed data are incomplete"
        )
    if not np.isin(treatment, [0.0, 1.0]).all():
        raise ValueError("Treatment must be binary")
    if not all(
        np.isfinite(values).all() for values in (features, outcomes, ground_truth)
    ):
        raise ValueError("Dataset contains NaN or infinite values")

    return DatasetBundle(
        name=normalized_name,
        features=features.astype(np.float32),
        treatment=treatment.astype(np.float32),
        outcomes=outcomes.astype(np.float32),
        ground_truth_ate=ground_truth.astype(np.float32),
    )


def split_indices(treatment: np.ndarray, seed: int) -> DatasetSplit:
    """Create the paper's stratified 70/20/10 train/validation/test split."""
    indices = np.arange(treatment.shape[0])
    train, remainder = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=treatment,
    )
    validation, test = train_test_split(
        remainder,
        test_size=1.0 / 3.0,
        random_state=seed,
        stratify=treatment[remainder],
    )
    return DatasetSplit(
        train=np.sort(train),
        validation=np.sort(validation),
        test=np.sort(test),
    )


def fit_normalization(
    bundle: DatasetBundle, train_indices: np.ndarray
) -> NormalizationStats:
    train_features = bundle.features[train_indices]
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_scale = train_features.std(axis=0, keepdims=True)
    feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)

    train_outcomes = bundle.outcomes[train_indices]
    outcome_mean = float(train_outcomes.mean())
    outcome_scale = float(train_outcomes.std())
    if outcome_scale < 1e-6:
        outcome_scale = 1.0

    return NormalizationStats(
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
        outcome_mean=outcome_mean,
        outcome_scale=outcome_scale,
    )


def stratified_subsample(
    bundle: DatasetBundle,
    max_samples: int | None,
    seed: int,
) -> DatasetBundle:
    """Select a deterministic, treatment-stratified subset for smoke tests."""
    if max_samples is None or max_samples >= bundle.treatment.shape[0]:
        return bundle
    if max_samples < 20:
        raise ValueError("max_samples must be at least 20")

    indices = np.arange(bundle.treatment.shape[0])
    selected, _ = train_test_split(
        indices,
        train_size=max_samples,
        random_state=seed,
        stratify=bundle.treatment,
    )
    selected = np.sort(selected)
    return DatasetBundle(
        name=bundle.name,
        features=bundle.features[selected],
        treatment=bundle.treatment[selected],
        outcomes=bundle.outcomes[selected],
        ground_truth_ate=bundle.ground_truth_ate,
    )
