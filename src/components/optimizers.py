from collections.abc import Iterator

import torch
from torch.nn.parameter import Parameter
from torch.optim import Optimizer


def get_optimizer(params: Iterator[Parameter]) -> Optimizer:
    return torch.optim.AdamW(params=params)
