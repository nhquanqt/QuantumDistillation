from __future__ import annotations

from dataclasses import dataclass

import numpy as onp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass
class DigitsSplits:
    train_x: onp.ndarray
    train_y: onp.ndarray
    val_x: onp.ndarray
    val_y: onp.ndarray
    test_x: onp.ndarray
    test_y: onp.ndarray


def _normalize_rows(features: onp.ndarray, eps: float = 1e-12) -> onp.ndarray:
    norms = onp.linalg.norm(features, axis=1, keepdims=True)
    norms = onp.maximum(norms, eps)
    return features / norms


def _one_hot(labels: onp.ndarray, num_classes: int = 10) -> onp.ndarray:
    encoded = onp.zeros((labels.shape[0], num_classes), dtype=onp.float64)
    encoded[onp.arange(labels.shape[0]), labels] = 1.0
    return encoded


def load_mnist8x8_splits(
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 123,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
) -> DigitsSplits:
    """Load the sklearn 8x8 handwritten digits dataset.

    The images are flattened to 64 features, scaled to [0, 1], and normalized
    so they can be embedded directly with amplitude embedding on 6 qubits.
    """

    digits = load_digits()
    features = digits.images.reshape((-1, 64)).astype(onp.float64) / 16.0
    features = _normalize_rows(features)
    labels = digits.target.astype(onp.int64)

    train_x, test_x, train_y_raw, test_y_raw = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_x, val_x, train_y_raw, val_y_raw = train_test_split(
        train_x,
        train_y_raw,
        test_size=val_size,
        random_state=seed,
        stratify=train_y_raw,
    )

    if train_limit is not None:
        train_x = train_x[:train_limit]
        train_y_raw = train_y_raw[:train_limit]
    if val_limit is not None:
        val_x = val_x[:val_limit]
        val_y_raw = val_y_raw[:val_limit]
    if test_limit is not None:
        test_x = test_x[:test_limit]
        test_y_raw = test_y_raw[:test_limit]

    return DigitsSplits(
        train_x=train_x,
        train_y=_one_hot(train_y_raw),
        val_x=val_x,
        val_y=_one_hot(val_y_raw),
        test_x=test_x,
        test_y=_one_hot(test_y_raw),
    )
