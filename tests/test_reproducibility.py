import torch

from src.config import ReproducibilityConfig
from src.pipeline import make_reproducibility


def test_make_reproducibility_toggles_deterministic_mode() -> None:
    cfg_strict = ReproducibilityConfig(deterministic=True, warn_only=True)
    make_reproducibility(cfg_strict)
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True

    cfg_fast = ReproducibilityConfig(deterministic=False)
    make_reproducibility(cfg_fast)
    assert torch.are_deterministic_algorithms_enabled() is False
