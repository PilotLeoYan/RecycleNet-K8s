from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.components.evaluator import Evaluator


def test_evaluator_test_model(tmp_path: Path) -> None:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 16 * 16, 2),
    )

    weights_path = tmp_path / "model_weights.pth"
    torch.save(model.state_dict(), weights_path)

    x = torch.randn(6, 3, 16, 16)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    test_loader = DataLoader(dataset, batch_size=2)

    evaluator = Evaluator(
        model=model,
        weights_path=weights_path,
        test_loader=test_loader,
        device="cpu",
    )

    predictions, probas, labels = evaluator._test_model()

    assert isinstance(predictions, np.ndarray)
    assert isinstance(probas, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(predictions) == 6
    assert probas.shape == (6, 2)
    assert len(labels) == 6


def test_evaluator_evaluate(tmp_path: Path) -> None:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 16 * 16, 2),
    )

    weights_path = tmp_path / "model_weights.pth"
    torch.save(model.state_dict(), weights_path)

    x = torch.randn(6, 3, 16, 16)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    test_loader = DataLoader(dataset, batch_size=2)

    evaluator = Evaluator(
        model=model,
        weights_path=weights_path,
        test_loader=test_loader,
        device="cpu",
    )

    # Should run end-to-end without throwing exceptions
    evaluator.evaluate()
