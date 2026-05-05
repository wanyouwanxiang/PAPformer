"""
Adaptive Multi-Scale Patch Transformer (PAPformer)

A Transformer-based time series forecasting model with adaptive patch tokenization.
The patch size is dynamically selected per token via a gating network conditioned
on local signal complexity (variance + finite-difference energy).

Reference:
    Paper: PAPformer: Adaptive Multi-Scale Patch Transformer for Time Series Forecasting
"""

import argparse
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from pandas import DataFrame, concat, read_csv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import Sequential

import keras

warnings.filterwarnings("ignore")


# ==============================================================================
# Metrics
# ==============================================================================


def custom_mape(y_true, y_pred):
    """Clip-based MAPE that avoids division by zero."""
    diff = tf.abs(y_true - y_pred)
    denom = tf.maximum(tf.abs(y_true), 1e-8)
    return tf.reduce_mean(diff / denom)


class R2ScoreCompat(Metric):
    """R2 score compatible with older Keras versions."""

    def __init__(self, name="r2_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name="ss_res", initializer="zeros")
        self.y_sum = self.add_weight(name="y_sum", initializer="zeros")
        self.y_sq_sum = self.add_weight(name="y_sq_sum", initializer="zeros")
        self.n = self.add_weight(name="n", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        self.ss_res.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.y_sum.assign_add(tf.reduce_sum(y_true))
        self.y_sq_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.n.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        y_mean = self.y_sum / (self.n + 1e-7)
        ss_tot = self.y_sq_sum - self.n * tf.square(y_mean)
        return 1.0 - self.ss_res / (ss_tot + 1e-7)

    def reset_state(self):
        self.ss_res.assign(0.0)
        self.y_sum.assign(0.0)
        self.y_sq_sum.assign(0.0)
        self.n.assign(0.0)


def _get_r2_metric():
    try:
        return keras.metrics.R2Score()
    except AttributeError:
        return R2ScoreCompat()


# ==============================================================================
# Data Utilities
# ==============================================================================


def series_to_supervised(data, n_in, n_out=1, dropnan=True):
    """Convert a time series into a supervised learning dataset via sliding window."""
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f"var{j + 1}(t-{i})" for j in range(n_vars)]

    for i in range(n_out):
        cols.append(df.shift(-i))
        suffix = "t" if i == 0 else f"t+{i}"
        names += [f"var{j + 1}({suffix})" for j in range(n_vars)]

    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# ==============================================================================
# Positional Encoding
# ==============================================================================


class PositionalEncoding(Layer):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        self.max_len = max_len
        self.d_model = d_model

    def call(self, x):
        n = tf.shape(x)[1]
        return x + self.pe[:, :n, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


# ==============================================================================
# Local Complexity Estimator
# ==============================================================================


class LocalComplexityEstimator(Layer):
    """Estimates local signal complexity via variance and finite-difference energy."""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, local_patch):
        mean = tf.reduce_mean(local_patch, axis=2)
        var = tf.math.reduce_variance(local_patch, axis=2)
        p = tf.shape(local_patch)[2]

        def compute_d1():
            d1 = local_patch[:, :, 1:, :] - local_patch[:, :, :-1, :]
            return tf.reduce_mean(tf.square(d1), axis=2)

        def zero_like_mean():
            return tf.zeros_like(mean)

        d1_energy = tf.cond(p >= 2, compute_d1, zero_like_mean)

        def compute_d2():
            d2 = (
                local_patch[:, :, 2:, :]
                - 2.0 * local_patch[:, :, 1:-1, :]
                + local_patch[:, :, :-2, :]
            )
            return tf.reduce_mean(tf.square(d2), axis=2)

        d2_energy = tf.cond(p >= 3, compute_d2, zero_like_mean)

        score = (
            tf.reduce_mean(var, axis=-1, keepdims=True)
            + tf.reduce_mean(d1_energy, axis=-1, keepdims=True)
            + tf.reduce_mean(d2_energy, axis=-1, keepdims=True)
        )
        feat = tf.concat([mean, var, d1_energy, d2_energy, score], axis=-1)
        return feat, score

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config


# ==============================================================================
# Adaptive Multi-Scale Patch Tokenizer
# ==============================================================================


class AdaptivePatchTokenizer(Layer):
    """Tokenizes input by extracting multi-scale patches and blending them via a
    gating network conditioned on local complexity."""

    def __init__(
        self,
        input_dim,
        d_model,
        patch_sizes,
        stride=1,
        gating_hidden=128,
        use_gumbel=False,
        gumbel_tau=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(patch_sizes) >= 2, "At least two patch sizes are required."
        self.input_dim = input_dim
        self.d_model = d_model
        self.patch_sizes = sorted(patch_sizes)
        self.stride = stride
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau
        self.gating_hidden = gating_hidden
        self.min_patch = min(self.patch_sizes)

        self.patch_embedders = {
            str(p): Dense(d_model, name=f"patch_embed_{p}")
            for p in self.patch_sizes
        }

        self.complexity_estimator = LocalComplexityEstimator(name="complexity_estimator")

        self.gating_net = Sequential(
            [
                Dense(gating_hidden, activation=tf.keras.activations.gelu),
                Dense(gating_hidden, activation=tf.keras.activations.gelu),
                Dense(len(self.patch_sizes)),
            ],
            name="gating_net",
        )

    def _compute_num_tokens(self, L):
        return math.ceil(max(L - self.min_patch, 0) / self.stride) + 1

    def _extract_patches(self, x, patch_size):
        input_shape = x.shape
        if input_shape[1] is None:
            raise ValueError("Sequence length must be statically known.")

        L = int(input_shape[1])
        C = int(input_shape[2])
        N = self._compute_num_tokens(L)

        last_start = (N - 1) * self.stride
        required_len = last_start + patch_size

        if required_len > L:
            pad_len = required_len - L
            pad = tf.repeat(x[:, -1:, :], repeats=pad_len, axis=1)
            x_pad = tf.concat([x, pad], axis=1)
        else:
            x_pad = x

        starts = tf.range(N, dtype=tf.int32) * self.stride
        offsets = tf.range(patch_size, dtype=tf.int32)
        idx = starts[:, None] + offsets[None, :]

        patches = tf.gather(x_pad, idx, axis=1)
        patches.set_shape([None, N, patch_size, C])
        return patches

    def _compute_prior(self, complexity_score):
        sizes = tf.constant(self.patch_sizes, dtype=complexity_score.dtype)
        sizes = tf.reshape(sizes, [1, 1, -1])
        logits = -1.0 * complexity_score * sizes
        return tf.nn.softmax(logits, axis=-1)

    def call(self, x, training=None):
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"input_dim mismatch: expected {self.input_dim}, got {x.shape[-1]}"
            )

        local_patch = self._extract_patches(x, self.min_patch)
        gate_feat, complexity_score = self.complexity_estimator(local_patch)
        gate_logits = self.gating_net(gate_feat, training=training)

        if self.use_gumbel and training:
            uniform = tf.random.uniform(
                tf.shape(gate_logits), minval=1e-6, maxval=1.0 - 1e-6
            )
            gumbel = -tf.math.log(-tf.math.log(uniform))
            gate_weights = tf.nn.softmax(
                (gate_logits + gumbel) / self.gumbel_tau, axis=-1
            )
        else:
            gate_weights = tf.nn.softmax(gate_logits, axis=-1)

        multi_scale_tokens = []
        for p in self.patch_sizes:
            patches = self._extract_patches(x, p)
            patches = tf.reshape(patches, [tf.shape(patches)[0], tf.shape(patches)[1], p * self.input_dim])
            token = self.patch_embedders[str(p)](patches)
            multi_scale_tokens.append(token)

        multi_scale_tokens = tf.stack(multi_scale_tokens, axis=2)
        tokens = tf.reduce_sum(
            tf.expand_dims(gate_weights, axis=-1) * multi_scale_tokens, axis=2
        )

        aux = {
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "gate_prior": self._compute_prior(complexity_score),
            "complexity_score": complexity_score,
        }
        return tokens, aux

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "d_model": self.d_model,
                "patch_sizes": self.patch_sizes,
                "stride": self.stride,
                "gating_hidden": self.gating_hidden,
                "use_gumbel": self.use_gumbel,
                "gumbel_tau": self.gumbel_tau,
            }
        )
        return config


# ==============================================================================
# Transformer Encoder Block
# ==============================================================================


class TransformerEncoderBlock(Layer):
    """Pre-LN Transformer encoder block with GELU activation in FFN."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead, dropout=dropout
        )
        self.dropout1 = Dropout(dropout)

        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.ffn_dense1 = Dense(dim_feedforward, activation=tf.keras.activations.gelu)
        self.ffn_dropout = Dropout(dropout)
        self.ffn_dense2 = Dense(d_model)
        self.dropout2 = Dropout(dropout)

    def call(self, x, training=None):
        h = self.norm1(x)
        h = self.attn(h, h, training=training)
        h = self.dropout1(h, training=training)
        x = x + h

        h2 = self.norm2(x)
        h2 = self.ffn_dense1(h2)
        h2 = self.ffn_dropout(h2, training=training)
        h2 = self.ffn_dense2(h2)
        h2 = self.dropout2(h2, training=training)
        return x + h2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "nhead": self.nhead,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
            }
        )
        return config


# ==============================================================================
# Regularization
# ==============================================================================


def adaptive_patch_regularization(aux, lambda_prior=1.0, lambda_smooth=1.0, lambda_entropy=0.0):
    """Compute KL prior, temporal smoothness, and entropy regularization losses."""
    gate_weights = aux["gate_weights"]
    gate_prior = aux["gate_prior"]
    eps = 1e-8

    kl = gate_weights * (tf.math.log(gate_weights + eps) - tf.math.log(gate_prior + eps))
    loss_prior = tf.reduce_mean(tf.reduce_sum(kl, axis=-1))

    if gate_weights.shape[1] is not None and gate_weights.shape[1] >= 2:
        loss_smooth = tf.reduce_mean(
            tf.square(gate_weights[:, 1:, :] - gate_weights[:, :-1, :])
        )
    else:
        seq_len = tf.shape(gate_weights)[1]
        loss_smooth = tf.cond(
            seq_len >= 2,
            lambda: tf.reduce_mean(
                tf.square(gate_weights[:, 1:, :] - gate_weights[:, :-1, :])
            ),
            lambda: tf.constant(0.0, dtype=gate_weights.dtype),
        )

    entropy = -tf.reduce_mean(
        tf.reduce_sum(gate_weights * tf.math.log(gate_weights + eps), axis=-1)
    )

    total = lambda_prior * loss_prior + lambda_smooth * loss_smooth + lambda_entropy * entropy
    return {
        "loss_prior": loss_prior,
        "loss_smooth": loss_smooth,
        "loss_entropy": entropy,
        "loss_reg": total,
    }


# ==============================================================================
# AdaptivePatchTransformer Model
# ==============================================================================


class AdaptivePatchTransformerModel(keras.Model):
    """Full model: adaptive patch tokenizer -> positional encoding -> Transformer encoder -> forecast head."""

    def __init__(
        self,
        input_dim,
        target_dim=1,
        pred_len=1,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.02,
        patch_sizes=None,
        stride=1,
        gating_hidden=128,
        use_gumbel=False,
        gumbel_tau=1.0,
        use_cls_token=False,
        lambda_prior=0.1,
        lambda_smooth=0.05,
        lambda_entropy=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if patch_sizes is None:
            patch_sizes = [2, 3, 5, 7]

        self.input_dim = input_dim
        self.target_dim = target_dim
        self.pred_len = pred_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.patch_sizes = patch_sizes
        self.stride = stride
        self.gating_hidden = gating_hidden
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau
        self.use_cls_token = use_cls_token
        self.lambda_prior = lambda_prior
        self.lambda_smooth = lambda_smooth
        self.lambda_entropy = lambda_entropy

        self.tokenizer = AdaptivePatchTokenizer(
            input_dim=input_dim,
            d_model=d_model,
            patch_sizes=patch_sizes,
            stride=stride,
            gating_hidden=gating_hidden,
            use_gumbel=use_gumbel,
            gumbel_tau=gumbel_tau,
            name="adaptive_patch_tokenizer",
        )

        if use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, d_model),
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
            )
        else:
            self.cls_token = None

        self.pos_encoder = PositionalEncoding(
            d_model=d_model, max_len=5000, name="positional_encoding"
        )
        self.encoder_layers = [
            TransformerEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                name=f"encoder_block_{i}",
            )
            for i in range(num_layers)
        ]
        self.norm = LayerNormalization(epsilon=1e-6, name="final_norm")

        self.head = Sequential(
            [
                Dense(d_model, activation=tf.keras.activations.gelu),
                Dropout(dropout),
                Dense(pred_len * target_dim, activation="linear"),
            ],
            name="forecast_head",
        )

        self.last_aux = None

    def call(self, x, training=None):
        tokens, aux = self.tokenizer(x, training=training)

        if self.use_cls_token:
            batch_size = tf.shape(x)[0]
            cls = tf.tile(self.cls_token, [batch_size, 1, 1])
            tokens = tf.concat([cls, tokens], axis=1)

        tokens = self.pos_encoder(tokens)

        h = tokens
        for layer in self.encoder_layers:
            h = layer(h, training=training)
        h = self.norm(h)

        pooled = h[:, 0, :] if self.use_cls_token else tf.reduce_mean(h, axis=1)
        out = self.head(pooled, training=training)
        y_hat = tf.reshape(out, [-1, self.pred_len, self.target_dim])

        self.last_aux = aux

        reg_losses = adaptive_patch_regularization(
            aux,
            lambda_prior=self.lambda_prior,
            lambda_smooth=self.lambda_smooth,
            lambda_entropy=self.lambda_entropy,
        )
        self.add_loss(reg_losses["loss_reg"])

        return tf.squeeze(y_hat, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "pred_len": self.pred_len,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout_rate,
                "patch_sizes": self.patch_sizes,
                "stride": self.stride,
                "gating_hidden": self.gating_hidden,
                "use_gumbel": self.use_gumbel,
                "gumbel_tau": self.gumbel_tau,
                "use_cls_token": self.use_cls_token,
                "lambda_prior": self.lambda_prior,
                "lambda_smooth": self.lambda_smooth,
                "lambda_entropy": self.lambda_entropy,
            }
        )
        return config


# ==============================================================================
# Model Builder
# ==============================================================================


def build_adaptive_patch_transformer(
    input_shape,
    target_dim=1,
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=256,
    dropout=0.02,
    patch_sizes=(2, 3, 5, 7, 11, 14),
    stride=1,
    gating_hidden=128,
    use_gumbel=False,
    gumbel_tau=1.0,
    use_cls_token=False,
    lambda_prior=0.1,
    lambda_smooth=0.05,
    lambda_entropy=0.001,
):
    """Build and initialize an AdaptivePatchTransformerModel."""
    model = AdaptivePatchTransformerModel(
        input_dim=input_shape[1],
        target_dim=target_dim,
        pred_len=1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        patch_sizes=list(patch_sizes),
        stride=stride,
        gating_hidden=gating_hidden,
        use_gumbel=use_gumbel,
        gumbel_tau=gumbel_tau,
        use_cls_token=use_cls_token,
        lambda_prior=lambda_prior,
        lambda_smooth=lambda_smooth,
        lambda_entropy=lambda_entropy,
        name="AdaptivePatchTransformer",
    )
    # Forward a dummy input to build all weights
    dummy = tf.zeros((1, input_shape[0], input_shape[1]), dtype=tf.float32)
    _ = model(dummy, training=False)
    return model


# ==============================================================================
# Training Pipeline
# ==============================================================================


def train_model(args):
    """Load data, build model, train, and evaluate."""
    dataset = read_csv(args.data_file, header=0, index_col=0, encoding="utf-16")
    if args.data_length is not None:
        dataset = dataset.iloc[: args.data_length].copy()

    dataset = dataset.astype("float32")
    target = dataset.values[:, 35].astype("float32").reshape(-1, 1)
    features = dataset.values[:, 0:13].astype("float32")

    total_samples = len(dataset)
    n_train = int(total_samples * 0.8)
    n_val = int(total_samples * 0.9)

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))

    train_X = scaler_X.fit_transform(features[:n_train])
    train_y = scaler_Y.fit_transform(target[:n_train])
    val_X = scaler_X.transform(features[n_train:n_val])
    val_y = scaler_Y.transform(target[n_train:n_val])
    test_X = scaler_X.transform(features[n_val:])
    test_y = scaler_Y.transform(target[n_val:])

    train_n = np.hstack((train_y, train_X))
    val_n = np.hstack((val_y, val_X))
    test_n = np.hstack((test_y, test_X))

    train_df = series_to_supervised(train_n, args.timestep, 1)
    val_df = series_to_supervised(val_n, args.timestep, 1)
    test_df = series_to_supervised(test_n, args.timestep, 1)

    n_vars = train_n.shape[1]
    input_cols = list(range(n_vars * args.timestep))
    output_cols = [n_vars * args.timestep]

    train_df = train_df.iloc[:, input_cols + output_cols]
    val_df = val_df.iloc[:, input_cols + output_cols]
    test_df = test_df.iloc[:, input_cols + output_cols]

    train_X, train_y = train_df.values[:, :-1], train_df.values[:, -1]
    val_X, val_y = val_df.values[:, :-1], val_df.values[:, -1]
    test_X, test_y = test_df.values[:, :-1], test_df.values[:, -1]

    train_X = train_X.reshape((-1, args.timestep, n_vars)).astype(np.float32)
    val_X = val_X.reshape((-1, args.timestep, n_vars)).astype(np.float32)
    test_X = test_X.reshape((-1, args.timestep, n_vars)).astype(np.float32)

    train_y = train_y.astype(np.float32)
    val_y = val_y.astype(np.float32)
    test_y = test_y.astype(np.float32)

    model = build_adaptive_patch_transformer(
        input_shape=(train_X.shape[1], train_X.shape[2]),
        target_dim=1,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        patch_sizes=args.patch_sizes,
        stride=1,
        gating_hidden=128,
        lambda_prior=args.lambda_prior,
        lambda_smooth=args.lambda_smooth,
        lambda_entropy=args.lambda_entropy,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(), custom_mape, _get_r2_metric()],
    )

    callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
    )

    history = model.fit(
        train_X,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[callback],
        validation_data=(val_X, val_y),
        verbose=2,
        shuffle=False,
    )

    # Save weights and scalers
    model.save_weights(args.save_weights)
    joblib.dump(scaler_X, args.save_scaler_x)
    joblib.dump(scaler_Y, args.save_scaler_y)

    # Save loss history
    loss_df = pd.DataFrame(
        {
            "epoch": range(1, len(history.history["loss"]) + 1),
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
        }
    )
    loss_df.to_csv(args.save_loss_csv, index=False)

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_loss_plot, dpi=150)
    plt.show()

    return model, history, scaler_X, scaler_Y, test_X, test_y


# ==============================================================================
# Evaluation
# ==============================================================================


def evaluate_model(model, scaler_Y, test_X, test_y, save_csv=None, save_plot=None):
    """Evaluate on test set and print metrics in original scale."""
    preds_scaled = model.predict(test_X, verbose=0)
    preds = scaler_Y.inverse_transform(preds_scaled.reshape(-1, 1))
    true_values = scaler_Y.inverse_transform(test_y.reshape(-1, 1))

    mse = mean_squared_error(true_values, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, preds)
    r2 = r2_score(true_values, preds)
    mape = np.mean(np.abs((true_values - preds) / (true_values + 1e-8))) * 100

    print("\n[Test Set Metrics - Original Scale]")
    print("=" * 50)
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R2:   {r2:.6f}")
    print("=" * 50)

    if save_csv:
        n_show = min(200000, len(test_X))
        df_eval = pd.DataFrame(
            {
                "true": true_values[:n_show].flatten(),
                "pred": preds[:n_show].flatten(),
                "error": (preds[:n_show] - true_values[:n_show]).flatten(),
                "relative_error(%)": (
                    (preds[:n_show] - true_values[:n_show])
                    / (true_values[:n_show] + 1e-8)
                    * 100
                ).flatten(),
            }
        )
        df_eval.to_csv(save_csv, index=False)
        print(f"Predictions saved to: {save_csv}")

    if save_plot:
        plt.figure(figsize=(12, 5))
        plt.plot(true_values, label="Ground Truth", alpha=0.8)
        plt.plot(preds, label="Prediction", alpha=0.8)
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_plot, dpi=150)
        plt.show()

    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train PAPformer on time series data")

    # Data
    parser.add_argument("--data_file", type=str, required=True, help="Path to CSV data file (UTF-16)")
    parser.add_argument("--data_length", type=int, default=None, help="Truncate dataset to this length")
    parser.add_argument("--timestep", type=int, default=14, help="Sliding window length")

    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--patience", type=int, default=100)

    # Model architecture
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--patch_sizes", type=int, nargs="+", default=[2, 3, 5, 7, 11, 14])

    # Regularization
    parser.add_argument("--lambda_prior", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_entropy", type=float, default=0.001)

    # Save paths
    parser.add_argument("--save_weights", type=str, default="papformer_weights.h5")
    parser.add_argument("--save_scaler_x", type=str, default="papformer_scaler_X.pkl")
    parser.add_argument("--save_scaler_y", type=str, default="papformer_scaler_Y.pkl")
    parser.add_argument("--save_loss_csv", type=str, default="papformer_loss_history.csv")
    parser.add_argument("--save_loss_plot", type=str, default="papformer_loss_curve.png")
    parser.add_argument("--save_pred_csv", type=str, default="papformer_predictions.csv")
    parser.add_argument("--save_pred_plot", type=str, default="papformer_predictions.png")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"GPUs available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

    model, history, scaler_X, scaler_Y, test_X, test_y = train_model(args)

    evaluate_model(
        model,
        scaler_Y,
        test_X,
        test_y,
        save_csv=args.save_pred_csv,
        save_plot=args.save_pred_plot,
    )
