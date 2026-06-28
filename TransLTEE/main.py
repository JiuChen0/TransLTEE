"""Command-line entry point for deterministic TransLTEE experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import ExperimentConfig
from .data import (
    fit_normalization,
    load_dataset,
    split_indices,
    stratified_subsample,
)
from .models import TransLTEE
from .train import evaluate_model, train_model
from .utils import set_global_seed, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALAR_METRICS = (
    "factual_mae",
    "factual_rmse",
    "ate_mae",
    "ate_rmse",
    "short_term_causal_effect_sign_accuracy",
    "long_term_causal_effect_sign_accuracy",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the double-head TransLTEE model."
    )
    parser.add_argument("--dataset", choices=("ihdp", "news", "both"), default="ihdp")
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of consecutive realizations, starting at --repetition.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--history-length", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dff", type=int, default=128)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--context-tokens", type=int, default=4)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Treatment-stratified subset for fast smoke tests.",
    )
    return parser.parse_args()


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    config.validate()
    run_seed = config.seed + config.repetition
    tf.keras.backend.clear_session()
    set_global_seed(run_seed)
    bundle = load_dataset(
        config.dataset,
        config.data_dir,
        repetition=config.repetition,
        sequence_length=config.sequence_length,
    )
    bundle = stratified_subsample(bundle, config.max_samples, run_seed)
    split = split_indices(bundle.treatment, run_seed)
    normalization = fit_normalization(bundle, split.train)

    model = TransLTEE(
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dff=config.dff,
        dropout_rate=config.dropout_rate,
        context_tokens=config.context_tokens,
        max_sequence_length=config.sequence_length,
        weight_decay=config.weight_decay,
    )
    history = train_model(model, bundle, split, normalization, config)
    metrics, predictions = evaluate_model(model, bundle, split, normalization, config)

    run_name = f"{config.dataset}_rep{config.repetition}_seed{run_seed}"
    run_directory = config.output_dir / run_name
    payload: dict[str, object] = {
        "config": config.to_dict(),
        "split_sizes": {
            "train": int(split.train.size),
            "validation": int(split.validation.size),
            "test": int(split.test.size),
        },
        "history": {
            "train_loss": history.train_loss,
            "validation_mse": history.validation_mse,
        },
        "metrics": metrics,
    }
    write_json(run_directory / "metrics.json", payload)
    run_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_directory / "predictions.npz", **predictions)
    model.save_weights(run_directory / "model.weights.h5")

    print(
        f"{run_name}: factual_rmse={metrics['factual_rmse']:.4f} "
        f"ate_mae={metrics['ate_mae']:.4f}"
    )
    return payload


def aggregate_runs(summaries: list[dict[str, object]]) -> dict[str, object]:
    """Compute per-dataset mean and standard deviation across realizations."""
    aggregate: dict[str, object] = {}
    dataset_names = sorted(
        {str(summary["config"]["dataset"]) for summary in summaries}  # type: ignore[index]
    )
    for dataset in dataset_names:
        dataset_runs = [
            summary
            for summary in summaries
            if summary["config"]["dataset"] == dataset  # type: ignore[index]
        ]
        aggregate[dataset] = {
            metric: {
                "mean": float(
                    np.mean([run["metrics"][metric] for run in dataset_runs])  # type: ignore[index]
                ),
                "std": float(
                    np.std([run["metrics"][metric] for run in dataset_runs])  # type: ignore[index]
                ),
            }
            for metric in SCALAR_METRICS
        }
    return aggregate


def main() -> None:
    arguments = parse_arguments()
    datasets = ("ihdp", "news") if arguments.dataset == "both" else (arguments.dataset,)
    end_repetition = arguments.repetition + arguments.repetitions
    if arguments.repetition < 1 or end_repetition > 11:
        raise ValueError(
            "Requested repetitions must stay within the committed range 1..10"
        )

    summaries = []
    for dataset in datasets:
        for repetition in range(arguments.repetition, end_repetition):
            config = ExperimentConfig(
                dataset=dataset,
                repetition=repetition,
                seed=arguments.seed,
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                learning_rate=arguments.learning_rate,
                gamma=arguments.gamma,
                weight_decay=arguments.weight_decay,
                sequence_length=arguments.sequence_length,
                history_length=arguments.history_length,
                d_model=arguments.d_model,
                num_layers=arguments.num_layers,
                num_heads=arguments.num_heads,
                dff=arguments.dff,
                dropout_rate=arguments.dropout_rate,
                context_tokens=arguments.context_tokens,
                sinkhorn_iterations=arguments.sinkhorn_iterations,
                sinkhorn_epsilon=arguments.sinkhorn_epsilon,
                data_dir=arguments.data_dir,
                output_dir=arguments.output_dir,
                max_samples=arguments.max_samples,
            )
            summaries.append(run_experiment(config))

    aggregate_path = arguments.output_dir / "summary.json"
    write_json(
        aggregate_path,
        {
            "aggregate": aggregate_runs(summaries),
            "runs": summaries,
        },
    )
    print(f"Wrote experiment summary to {aggregate_path}")


if __name__ == "__main__":
    main()
