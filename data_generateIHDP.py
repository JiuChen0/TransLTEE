"""Deterministically generate longitudinal IHDP outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent


def generate_ihdp(
    data_dir: Path,
    repetition: int,
    seed: int,
    timesteps: int = 100,
) -> None:
    dataset_dir = data_dir / "IHDP"
    raw = np.loadtxt(
        dataset_dir / "csv" / f"ihdp_npci_{repetition}.csv",
        delimiter=",",
        dtype=np.float64,
    )
    treatment = raw[:, 0]
    factual_initial = raw[:, 1]
    counterfactual_initial = raw[:, 2]
    features = raw[:, 5:]
    sample_count = treatment.shape[0]

    potential_control = np.zeros((sample_count, timesteps), dtype=np.float64)
    potential_treated = np.zeros_like(potential_control)
    potential_control[:, 0] = np.where(
        treatment > 0.5,
        counterfactual_initial,
        factual_initial,
    )
    potential_treated[:, 0] = np.where(
        treatment > 0.5,
        factual_initial,
        counterfactual_initial,
    )

    rng = np.random.default_rng(seed)
    beta_values = np.array([0, 1, 2, 3, 4])
    beta_probabilities = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
    for timestep in range(1, timesteps):
        beta = rng.choice(
            beta_values,
            size=features.shape[1],
            p=beta_probabilities,
        )
        noise = rng.normal(0.0, 1.0, sample_count)
        baseline = features @ beta
        control_history = 0.02 * potential_control[:, :timestep].sum(axis=1)
        treated_history = 0.02 * potential_treated[:, :timestep].sum(axis=1)
        potential_control[:, timestep] = baseline + control_history + noise
        potential_treated[:, timestep] = baseline + 4.0 + treated_history + noise

    factual = np.where(
        treatment[:, np.newaxis] > 0.5,
        potential_treated,
        potential_control,
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
        generate_ihdp(
            arguments.data_dir,
            repetition,
            seed=arguments.seed + repetition,
            timesteps=arguments.timesteps,
        )
        print(f"Generated IHDP repetition {repetition}")


if __name__ == "__main__":
    main()
