"""Training and evaluation routines for TransLTEE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .config import ExperimentConfig
from .data import DatasetBundle, DatasetSplit, NormalizationStats
from .models import TransLTEE


@dataclass(frozen=True)
class TrainingHistory:
    train_loss: list[float]
    validation_mse: list[float]


def make_decoder_input(outcomes: tf.Tensor) -> tf.Tensor:
    """Shift an outcome sequence right and prepend the autoregressive start token."""
    start = tf.zeros_like(outcomes[:, :1, :])
    return tf.concat([start, outcomes[:, :-1, :]], axis=1)


def select_factual_predictions(
    control_predictions: tf.Tensor,
    treatment_predictions: tf.Tensor,
    treatment: tf.Tensor,
) -> tf.Tensor:
    """Select a head without changing sample order."""
    mask = tf.reshape(tf.cast(treatment > 0.5, tf.bool), (-1, 1, 1))
    return tf.where(mask, treatment_predictions, control_predictions)


def sinkhorn_wasserstein(
    control: tf.Tensor,
    treated: tf.Tensor,
    iterations: int = 20,
    epsilon: float = 0.1,
) -> tf.Tensor:
    """Differentiable entropy-regularized approximation of Wasserstein-1."""
    differences = control[:, tf.newaxis, :] - treated[tf.newaxis, :, :]
    cost = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(differences), axis=-1), 1e-12))
    scale = tf.stop_gradient(tf.maximum(tf.reduce_mean(cost), 1e-6))
    kernel = tf.exp(-cost / (epsilon * scale + 1e-8))

    control_count = tf.shape(control)[0]
    treated_count = tf.shape(treated)[0]
    a = tf.fill((control_count,), 1.0 / tf.cast(control_count, tf.float32))
    b = tf.fill((treated_count,), 1.0 / tf.cast(treated_count, tf.float32))
    u = tf.ones_like(a)
    v = tf.ones_like(b)

    for _ in range(iterations):
        u = a / (tf.linalg.matvec(kernel, v) + 1e-8)
        v = b / (tf.linalg.matvec(kernel, u, transpose_a=True) + 1e-8)

    transport = u[:, tf.newaxis] * kernel * v[tf.newaxis, :]
    return tf.reduce_sum(transport * cost)


def representation_balance_loss(
    control_representation: tf.Tensor,
    treatment_representation: tf.Tensor,
    treatment: tf.Tensor,
    iterations: int,
    epsilon: float,
) -> tf.Tensor:
    """Compare assigned control and treated representations for one mini-batch."""
    treatment = tf.reshape(treatment, (-1,))
    control_mask = treatment < 0.5
    treated_mask = treatment > 0.5
    control = tf.boolean_mask(
        tf.reduce_mean(control_representation, axis=1),
        control_mask,
    )
    treated = tf.boolean_mask(
        tf.reduce_mean(treatment_representation, axis=1),
        treated_mask,
    )

    has_both_groups = tf.logical_and(tf.shape(control)[0] > 0, tf.shape(treated)[0] > 0)
    return tf.cond(
        has_both_groups,
        lambda: sinkhorn_wasserstein(control, treated, iterations, epsilon),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


def _batches(
    features: np.ndarray,
    treatment: np.ndarray,
    outcomes: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            features.astype(np.float32),
            treatment.astype(np.float32),
            outcomes[..., np.newaxis].astype(np.float32),
        )
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=features.shape[0],
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_model(
    model: TransLTEE,
    bundle: DatasetBundle,
    split: DatasetSplit,
    normalization: NormalizationStats,
    config: ExperimentConfig,
) -> TrainingHistory:
    features = normalization.transform_features(bundle.features)
    outcomes = normalization.transform_outcomes(bundle.outcomes)
    train_data = _batches(
        features[split.train],
        bundle.treatment[split.train],
        outcomes[split.train],
        config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_data = _batches(
        features[split.validation],
        bundle.treatment[split.validation],
        outcomes[split.validation],
        config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    train_history: list[float] = []
    validation_history: list[float] = []

    for epoch in range(config.epochs):
        train_metric = tf.keras.metrics.Mean()
        for batch_features, batch_treatment, batch_outcomes in train_data:
            decoder_input = make_decoder_input(batch_outcomes)
            with tf.GradientTape() as tape:
                control, treated, control_rep, treated_rep = model(
                    (batch_features, decoder_input, decoder_input),
                    training=True,
                )
                factual = select_factual_predictions(control, treated, batch_treatment)
                prediction_loss = tf.reduce_mean(tf.square(factual - batch_outcomes))
                balance_loss = representation_balance_loss(
                    control_rep,
                    treated_rep,
                    batch_treatment,
                    config.sinkhorn_iterations,
                    config.sinkhorn_epsilon,
                )
                regularization = (
                    tf.add_n(model.losses)
                    if model.losses
                    else tf.constant(0.0, dtype=tf.float32)
                )
                total_loss = (
                    prediction_loss + config.gamma * balance_loss + regularization
                )

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                (gradient, variable)
                for gradient, variable in zip(gradients, model.trainable_variables)
                if gradient is not None
            )
            train_metric.update_state(total_loss)

        validation_metric = tf.keras.metrics.Mean()
        for batch_features, batch_treatment, batch_outcomes in validation_data:
            decoder_input = make_decoder_input(batch_outcomes)
            control, treated, _, _ = model(
                (batch_features, decoder_input, decoder_input),
                training=False,
            )
            factual = select_factual_predictions(control, treated, batch_treatment)
            validation_metric.update_state(
                tf.reduce_mean(tf.square(factual - batch_outcomes))
            )

        train_value = float(train_metric.result().numpy())
        validation_value = float(validation_metric.result().numpy())
        train_history.append(train_value)
        validation_history.append(validation_value)
        print(
            f"Epoch {epoch + 1:03d}/{config.epochs:03d} "
            f"loss={train_value:.6f} val_mse={validation_value:.6f}"
        )

    return TrainingHistory(
        train_loss=train_history,
        validation_mse=validation_history,
    )


def _rollout_batches(
    model: TransLTEE,
    features: np.ndarray,
    batch_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    control_predictions = []
    treatment_predictions = []
    dataset = tf.data.Dataset.from_tensor_slices(features.astype(np.float32)).batch(
        batch_size
    )
    for batch_features in dataset:
        control, treated = model.rollout(batch_features, horizon)
        control_predictions.append(tf.squeeze(control, axis=-1).numpy())
        treatment_predictions.append(tf.squeeze(treated, axis=-1).numpy())
    return np.concatenate(control_predictions), np.concatenate(treatment_predictions)


def evaluate_model(
    model: TransLTEE,
    bundle: DatasetBundle,
    split: DatasetSplit,
    normalization: NormalizationStats,
    config: ExperimentConfig,
) -> tuple[dict[str, float | list[float]], dict[str, np.ndarray]]:
    features = normalization.transform_features(bundle.features[split.test])
    control_scaled, treated_scaled = _rollout_batches(
        model,
        features,
        config.batch_size,
        config.sequence_length,
    )
    control = normalization.inverse_outcomes(control_scaled)
    treated = normalization.inverse_outcomes(treated_scaled)
    treatment = bundle.treatment[split.test, np.newaxis]
    factual = np.where(treatment > 0.5, treated, control)
    observed = bundle.outcomes[split.test]

    predicted_ate = np.mean(treated - control, axis=0)
    ground_truth = bundle.ground_truth_ate
    ate_error = predicted_ate - ground_truth
    sign_match = np.sign(predicted_ate) == np.sign(ground_truth)
    short_term = slice(0, config.history_length)
    long_term = slice(config.history_length, config.sequence_length)

    metrics: dict[str, float | list[float]] = {
        "factual_mae": float(np.mean(np.abs(factual - observed))),
        "factual_rmse": float(np.sqrt(np.mean(np.square(factual - observed)))),
        "ate_mae": float(np.mean(np.abs(ate_error))),
        "ate_rmse": float(np.sqrt(np.mean(np.square(ate_error)))),
        "short_term_causal_effect_sign_accuracy": float(
            100.0 * np.mean(sign_match[short_term])
        ),
        "long_term_causal_effect_sign_accuracy": float(
            100.0 * np.mean(sign_match[long_term])
        ),
        "predicted_ate": predicted_ate.astype(float).tolist(),
        "ground_truth_ate": ground_truth.astype(float).tolist(),
    }
    predictions = {
        "test_indices": split.test,
        "control": control,
        "treated": treated,
        "factual": factual,
        "observed": observed,
        "predicted_ate": predicted_ate,
        "ground_truth_ate": ground_truth,
    }
    return metrics, predictions
