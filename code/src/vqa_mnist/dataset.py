from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST


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


def _downsample_and_flatten(images: torch.Tensor) -> np.ndarray:
    resized = F.interpolate(
        images.unsqueeze(1),
        size=(8, 8),
        mode="bilinear",
        align_corners=False,
    )
    flattened = resized.reshape(resized.shape[0], 64).cpu().numpy().astype(np.float64)
    return _normalize_rows(flattened)


def _split_train_val(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []

    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0].copy()
        rng.shuffle(label_indices)
        val_count = max(1, int(round(len(label_indices) * val_size)))
        val_indices.append(label_indices[:val_count])
        train_indices.append(label_indices[val_count:])

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (
        features[train_idx],
        features[val_idx],
        labels[train_idx],
        labels[val_idx],
    )


def load_mnist8x8_splits(
    *,
    num_classes: int = 10,
    val_size: float = 0.1,
    seed: int = 123,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
) -> DigitsSplits:
    """Load torchvision MNIST and resize images to 8x8.

    The official MNIST train/test split is preserved. Images are scaled to
    [0, 1], resized from 28x28 to 8x8, flattened to 64 features, and then
    normalized so they can be embedded directly with amplitude embedding on
    6 qubits.
    """

    if num_classes not in (4, 10):
        raise ValueError(f"Unsupported num_classes: {num_classes}")

    data_root = Path(__file__).resolve().parents[2] / "data"
    train_dataset = MNIST(root=data_root, train=True, download=True)
    test_dataset = MNIST(root=data_root, train=False, download=True)

    train_images = train_dataset.data.to(torch.float32) / 255.0
    train_labels_all = train_dataset.targets.cpu().numpy().astype(np.int64)
    test_images = test_dataset.data.to(torch.float32) / 255.0
    test_labels_raw = test_dataset.targets.cpu().numpy().astype(np.int64)

    if num_classes == 4:
        train_mask = train_labels_all < 4
        test_mask = test_labels_raw < 4
        train_images = train_images[train_mask]
        train_labels_all = train_labels_all[train_mask]
        test_images = test_images[test_mask]
        test_labels_raw = test_labels_raw[test_mask]

    train_features_all = _downsample_and_flatten(train_images)
    test_x = _downsample_and_flatten(test_images)

    train_x, val_x, train_y_raw, val_y_raw = _split_train_val(
        train_features_all,
        train_labels_all,
        val_size=val_size,
        seed=seed,
    )

    if train_limit is not None:
        train_x = train_x[:train_limit]
        train_y_raw = train_y_raw[:train_limit]
    if val_limit is not None:
        val_x = val_x[:val_limit]
        val_y_raw = val_y_raw[:val_limit]
    if test_limit is not None:
        test_x = test_x[:test_limit]
        test_labels_raw = test_labels_raw[:test_limit]

    return DigitsSplits(
        train_x=train_x,
        train_y=_one_hot(train_y_raw, num_classes=num_classes),
        val_x=val_x,
        val_y=_one_hot(val_y_raw, num_classes=num_classes),
        test_x=test_x,
        test_y=_one_hot(test_labels_raw, num_classes=num_classes),
    )
