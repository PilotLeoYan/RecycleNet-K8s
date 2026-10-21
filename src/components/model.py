"""Model architecture builder module for image classification."""

from typing import Literal, get_args

from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    MobileNetV3,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

MobileNetV3Size = Literal["small", "large"]
SUPPORTED_MOBILENET_V3_SIZES: tuple[str, ...] = get_args(MobileNetV3Size)


def build_mobilenet_v3(
    num_classes: int,
    freeze_base: bool = True,
    size: MobileNetV3Size = "small",
) -> MobileNetV3:
    """Construct a pre-trained MobileNetV3 architecture for classification.

    Loads the default ImageNet pre-trained weights for the requested size,
    freezes backbone convolutional layers if requested, and replaces the final
    linear layer with a new trainable head matching `num_classes`.

    Args:
        num_classes: Number of target output classes (e.g., 6 for TrashNet).
        freeze_base: Whether to freeze feature extractor weights for transfer learning.
        size: MobileNetV3 variant to build, either 'small' or 'large'.

    Returns:
        MobileNetV3: Configured PyTorch neural network model.

    Raises:
        ValueError: If `size` is not a supported MobileNetV3 variant.

    Example:
        ```python
        from src.components.model import build_mobilenet_v3

        model = build_mobilenet_v3(num_classes=6, freeze_base=True, size="large")
        ```
    """
    if size == "small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    elif size == "large":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    else:
        raise ValueError(
            f"Unsupported MobileNetV3 size '{size}'. "
            f"Expected one of {SUPPORTED_MOBILENET_V3_SIZES}."
        )

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
