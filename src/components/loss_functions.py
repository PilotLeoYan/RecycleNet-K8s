"""Loss function factory module for model training."""

import torch
from torch import nn


def get_criterion() -> nn.Module:
    """Instantiate and return the default loss function criterion.

    Returns:
        nn.Module: CrossEntropyLoss module for multi-class classification.
    """
    return torch.nn.CrossEntropyLoss()
