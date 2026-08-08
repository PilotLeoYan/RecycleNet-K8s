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

    in_features = model.classifier[0].in_features
    out_features = model.classifier[0].out_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(out_features, num_classes, bias=True),
    )
    return model
