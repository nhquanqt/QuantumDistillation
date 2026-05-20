from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass
class DigitsSplits:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def _normalize_rows(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


def _one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
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
    features = digits.images.reshape((-1, 64)).astype(np.float64) / 16.0
    features = _normalize_rows(features)
    labels = digits.target.astype(np.int64)

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
