import torch
from torch import nn


def get_criterion() -> nn.Module:
    """"""
    return torch.nn.CrossEntropyLoss()
