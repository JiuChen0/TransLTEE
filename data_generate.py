"""Deterministically generate longitudinal NEWS outcomes.

The committed ``.csv.x`` and ``.csv.y`` files are the static NEWS benchmark.
This script converts them into the 100-step semi-synthetic sequences used by
TransLTEE. The ``.y`` suffix is inherited from the upstream dataset; it is a
numeric CSV file, not Yacc source code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from TransLTEE.data import _load_news_features


PROJECT_ROOT = Path(__file__).resolve().parent


def generate_news(
    data_dir: Path,
    repetition: int,
    seed: int,
    timesteps: int = 100,
) -> None:
    dataset_dir = data_dir / "NEWS"
    labels_path = dataset_dir / f"topic_doc_mean_n5000_k3477_seed_{repetition}.csv.y"
    features_path = dataset_dir / f"topic_doc_mean_n5000_k3477_seed_{repetition}.csv.x"
    labels = np.loadtxt(labels_path, delimiter=",", dtype=np.float64)
    features = _load_news_features(features_path).astype(np.float64)
    treatment = labels[:, 0]
    factual_initial = labels[:, 1]
    counterfactual_initial = labels[:, 2]
    sample_count = treatment.shape[0]

    rng = np.random.default_rng(seed)
    topics = LatentDirichletAllocation(
        n_components=50,
        max_iter=5,
        learning_method="online",
        learning_offset=50.0,
        random_state=seed,
    ).fit_transform(features)
    treated_centroid = topics[rng.integers(sample_count)]
    population_centroid = topics.mean(axis=0)
    baseline = 50.0 * (topics @ population_centroid)
    treatment_effect = 50.0 * (topics @ treated_centroid)

    factual = np.zeros((sample_count, timesteps), dtype=np.float64)
    counterfactual = np.zeros_like(factual)
    factual[:, 0] = factual_initial
    counterfactual[:, 0] = counterfactual_initial

    for timestep in range(1, timesteps):
        # Common random numbers isolate treatment effects while retaining
        # independent observation noise for each unit.
        noise = rng.normal(0.0, 1.0, sample_count)
        factual_history = 0.03 * factual[:, :timestep].sum(axis=1)
        counterfactual_history = 0.03 * counterfactual[:, :timestep].sum(axis=1)
        factual[:, timestep] = (
            baseline + treatment * treatment_effect + factual_history + noise
        )
        counterfactual[:, timestep] = (
            baseline
            + (1.0 - treatment) * treatment_effect
            + counterfactual_history
            + noise
        )

    potential_treated = np.where(
        treatment[:, np.newaxis] > 0.5,
        factual,
        counterfactual,
    )
    potential_control = np.where(
        treatment[:, np.newaxis] > 0.5,
        counterfactual,
        factual,
    )
    ground_truth = np.mean(potential_treated - potential_control, axis=0)
    series = np.column_stack([treatment, factual])
    np.savetxt(
        dataset_dir / f"Series_groundtruth_{repetition}.txt",
        ground_truth,
        delimiter=",",
        fmt="%.6f",
    )
    np.savetxt(
        dataset_dir / f"Series_y_{repetition}.txt",
        series,
        delimiter=",",
        fmt="%.6f",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--timesteps", type=int, default=100)
    arguments = parser.parse_args()

    end = arguments.repetition + arguments.repetitions
    if arguments.repetition < 1 or end > 11:
        raise ValueError("Requested repetitions must stay within 1..10")
    for repetition in range(arguments.repetition, end):
        generate_news(
            arguments.data_dir,
            repetition,
            seed=arguments.seed + repetition,
            timesteps=arguments.timesteps,
        )
        print(f"Generated NEWS repetition {repetition}")


if __name__ == "__main__":
    main()
