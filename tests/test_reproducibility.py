"""Tests for deterministic random seed configuration and reproducibility utilities."""

import os
from unittest.mock import MagicMock, patch

import torch

from src.config import ReproducibilityConfig
from src.pipeline import make_reproducibility


def test_make_reproducibility_toggles_deterministic_mode() -> None:
    """Verify deterministic mode toggles deterministic algorithm flags."""
    cfg_strict = ReproducibilityConfig(deterministic=True, warn_only=True)
    make_reproducibility(cfg_strict)
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True

    cfg_fast = ReproducibilityConfig(deterministic=False)
    make_reproducibility(cfg_fast)
    assert torch.are_deterministic_algorithms_enabled() is False


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.manual_seed")
@patch("torch.cuda.manual_seed_all")
def test_make_reproducibility_cuda_deterministic(
    mock_seed_all: MagicMock,
    mock_seed: MagicMock,
    mock_cuda_avail: MagicMock,
) -> None:
    """Verify CUDA seed initialization and cuDNN flags in deterministic mode."""
    cfg = ReproducibilityConfig(
        random_seed=42,
        numpy_seed=42,
        torch_seed=42,
        deterministic=True,
        warn_only=True,
    )
    make_reproducibility(cfg)

    mock_cuda_avail.assert_called_once()
    mock_seed.assert_called_once_with(42)
    assert mock_seed_all.call_count >= 1
    mock_seed_all.assert_called_with(42)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.manual_seed")
@patch("torch.cuda.manual_seed_all")
def test_make_reproducibility_cuda_non_deterministic(
    mock_seed_all: MagicMock,
    mock_seed: MagicMock,
    mock_cuda_avail: MagicMock,
) -> None:
    """Verify CUDA seed initialization and cuDNN flags in benchmark mode."""
    cfg = ReproducibilityConfig(
        random_seed=123,
        numpy_seed=123,
        torch_seed=123,
        deterministic=False,
    )
    make_reproducibility(cfg)

    mock_cuda_avail.assert_called_once()
    mock_seed.assert_called_once_with(123)
    assert mock_seed_all.call_count >= 1
    mock_seed_all.assert_called_with(123)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    assert torch.are_deterministic_algorithms_enabled() is False
