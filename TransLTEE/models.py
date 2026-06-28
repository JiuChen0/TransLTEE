"""Double-head autoregressive Transformer used by TransLTEE."""

from __future__ import annotations

import tensorflow as tf


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float):
        super().__init__()
        key_dim = d_model // num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
        )
        self.feed_forward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation=tf.nn.gelu),
                tf.keras.layers.Dense(d_model),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, values: tf.Tensor, training: bool = False) -> tf.Tensor:
        attended = self.attention(
            query=values,
            value=values,
            key=values,
            training=training,
        )
        values = self.norm1(values + self.dropout1(attended, training=training))
        forwarded = self.feed_forward(values, training=training)
        return self.norm2(values + self.dropout2(forwarded, training=training))


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float):
        super().__init__()
        key_dim = d_model // num_heads
        self.causal_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
        )
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
        )
        self.feed_forward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation=tf.nn.gelu),
                tf.keras.layers.Dense(d_model),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(
        self,
        values: tf.Tensor,
        encoded_context: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        attended = self.causal_attention(
            query=values,
            value=values,
            key=values,
            use_causal_mask=True,
            training=training,
        )
        values = self.norm1(values + self.dropout1(attended, training=training))

        crossed = self.cross_attention(
            query=values,
            value=encoded_context,
            key=encoded_context,
            training=training,
        )
        values = self.norm2(values + self.dropout2(crossed, training=training))
        forwarded = self.feed_forward(values, training=training)
        return self.norm3(values + self.dropout3(forwarded, training=training))


class TreatmentHead(tf.keras.layers.Layer):
    """One independent Transformer encoder-decoder treatment head."""

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dff: int,
        dropout_rate: float,
        context_tokens: int,
        max_sequence_length: int,
        weight_decay: float,
        name: str,
    ):
        super().__init__(name=name)
        regularizer = tf.keras.regularizers.L2(weight_decay)
        self.context_position = tf.keras.layers.Embedding(context_tokens, d_model)
        self.outcome_projection = tf.keras.layers.Dense(
            d_model,
            kernel_regularizer=regularizer,
        )
        self.outcome_position = tf.keras.layers.Embedding(max_sequence_length, d_model)
        self.encoders = [
            TransformerEncoderBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.decoders = [
            TransformerDecoderBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(
            1,
            kernel_regularizer=regularizer,
        )

    def call(
        self,
        context: tf.Tensor,
        decoder_input: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        context_length = tf.shape(context)[1]
        context_positions = tf.range(context_length)
        encoded = context + self.context_position(context_positions)[tf.newaxis, :, :]
        for encoder in self.encoders:
            encoded = encoder(encoded, training=training)

        sequence_length = tf.shape(decoder_input)[1]
        sequence_positions = tf.range(sequence_length)
        decoded = self.outcome_projection(decoder_input)
        decoded += self.outcome_position(sequence_positions)[tf.newaxis, :, :]
        for decoder in self.decoders:
            decoded = decoder(decoded, encoded, training=training)
        return self.output_layer(decoded), encoded


class TransLTEE(tf.keras.Model):
    """Shared context projection followed by independent control/treatment heads."""

    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dff: int = 128,
        dropout_rate: float = 0.1,
        context_tokens: int = 4,
        max_sequence_length: int = 100,
        weight_decay: float = 1e-6,
    ):
        super().__init__(name="transltee")
        self.max_sequence_length = max_sequence_length
        self.context_tokens = context_tokens
        regularizer = tf.keras.regularizers.L2(weight_decay)
        self.context_projection = tf.keras.layers.Dense(
            context_tokens * d_model,
            activation=tf.nn.gelu,
            kernel_regularizer=regularizer,
            name="phi",
        )
        head_arguments = dict(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            context_tokens=context_tokens,
            max_sequence_length=max_sequence_length,
            weight_decay=weight_decay,
        )
        self.control_head = TreatmentHead(**head_arguments, name="control_head")
        self.treatment_head = TreatmentHead(**head_arguments, name="treatment_head")
        self.d_model = d_model

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        features, control_decoder_input, treatment_decoder_input = inputs
        projected = self.context_projection(features)
        context = tf.reshape(
            projected,
            (-1, self.context_tokens, self.d_model),
        )
        control_output, control_representation = self.control_head(
            context,
            control_decoder_input,
            training=training,
        )
        treatment_output, treatment_representation = self.treatment_head(
            context,
            treatment_decoder_input,
            training=training,
        )
        return (
            control_output,
            treatment_output,
            control_representation,
            treatment_representation,
        )

    def rollout(
        self,
        features: tf.Tensor,
        horizon: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Autoregressively predict both potential-outcome sequences."""
        if horizon > self.max_sequence_length:
            raise ValueError("horizon exceeds the configured maximum sequence length")

        batch_size = tf.shape(features)[0]
        control_input = tf.zeros((batch_size, 1, 1), dtype=tf.float32)
        treatment_input = tf.zeros((batch_size, 1, 1), dtype=tf.float32)
        control_steps = []
        treatment_steps = []

        for step in range(horizon):
            control, treatment, _, _ = self(
                (features, control_input, treatment_input),
                training=False,
            )
            next_control = control[:, -1:, :]
            next_treatment = treatment[:, -1:, :]
            control_steps.append(next_control)
            treatment_steps.append(next_treatment)
            if step + 1 < horizon:
                control_input = tf.concat([control_input, next_control], axis=1)
                treatment_input = tf.concat([treatment_input, next_treatment], axis=1)

        return tf.concat(control_steps, axis=1), tf.concat(treatment_steps, axis=1)


# Backward-compatible import name used by early versions of the repository.
MyModel = TransLTEE
