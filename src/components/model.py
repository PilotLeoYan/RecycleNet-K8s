from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNetV3,
    mobilenet_v3_small,
)


def build_mobilenet_v3(num_classes: int, freeze_base: bool = True) -> MobileNetV3:
    """"""
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # only reaplce the last layer
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
