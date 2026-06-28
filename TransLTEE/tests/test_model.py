import numpy as np
import tensorflow as tf

from TransLTEE.models import TransLTEE
from TransLTEE.train import (
    make_decoder_input,
    select_factual_predictions,
    sinkhorn_wasserstein,
)


def test_double_head_shapes_and_sample_order():
    model = TransLTEE(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dff=16,
        dropout_rate=0.0,
        context_tokens=2,
        max_sequence_length=4,
    )
    features = tf.random.normal((6, 5), seed=1)
    outcomes = tf.random.normal((6, 4, 1), seed=2)
    decoder_input = make_decoder_input(outcomes)
    control, treated, control_rep, treated_rep = model(
        (features, decoder_input, decoder_input),
        training=False,
    )
    assert control.shape == treated.shape == (6, 4, 1)
    assert control_rep.shape == treated_rep.shape == (6, 2, 8)
    assert model.control_head is not model.treatment_head

    treatment = tf.constant([0, 1, 0, 1, 0, 1], dtype=tf.float32)
    factual = select_factual_predictions(control, treated, treatment)
    np.testing.assert_allclose(factual[0], control[0])
    np.testing.assert_allclose(factual[1], treated[1])


def test_rollout_and_sinkhorn_are_finite():
    model = TransLTEE(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dff=16,
        dropout_rate=0.0,
        context_tokens=2,
        max_sequence_length=4,
    )
    control, treated = model.rollout(tf.ones((3, 5)), horizon=4)
    assert control.shape == treated.shape == (3, 4, 1)
    assert np.isfinite(control.numpy()).all()
    distance = sinkhorn_wasserstein(
        tf.constant([[0.0], [1.0]]),
        tf.constant([[2.0], [3.0]]),
        iterations=10,
    )
    assert np.isfinite(float(distance.numpy()))
    assert float(distance.numpy()) > 0
