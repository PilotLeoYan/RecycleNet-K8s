"""Model architecture builder module for image classification."""

from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNetV3,
    mobilenet_v3_small,
)


def build_mobilenet_v3(num_classes: int, freeze_base: bool = True) -> MobileNetV3:
    """Constructs a pre-trained MobileNetV3-Small architecture for classification.

    Loads the default ImageNet pre-trained weights, freezes backbone convolutional
    layers if requested, and replaces the final linear layer with a new trainable
    head matching `num_classes`.

    Args:
        num_classes: Number of target output classes (e.g., 6 for TrashNet).
        freeze_base: Whether to freeze feature extractor weights for transfer learning.

    Returns:
        MobileNetV3: Configured PyTorch neural network model.

    Example:
        ```python
        from src.components.model import build_mobilenet_v3

        model = build_mobilenet_v3(num_classes=6, freeze_base=True)
        ```
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # replace the last layer
    last_in_features: int = model.classifier[-1].in_features  # type: ignore
    model.classifier[-1] = nn.Linear(
        in_features=last_in_features,
        out_features=num_classes,
        bias=True,
    )

    # ensure that new last layer is trainable
    for param in model.classifier[-1].parameters():
        param.requires_grad = True
    return model
