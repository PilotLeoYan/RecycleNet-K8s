import torch

from src.components.model import build_mobilenet_v3


def test_build_mobilenet_v3_structure() -> None:
    num_classes = 6
    model = build_mobilenet_v3(num_classes=num_classes, freeze_base=True)

    # Verify last classification layer
    last_layer = model.classifier[-1]
    assert last_layer.out_features == num_classes
    assert last_layer.weight.requires_grad is True

    # Verify base layers are frozen
    base_params = list(model.features.parameters())
    for param in base_params:
        assert param.requires_grad is False


def test_build_mobilenet_v3_forward() -> None:
    num_classes = 4
    model = build_mobilenet_v3(num_classes=num_classes, freeze_base=False)
    model.eval()

    # Simulated input tensor [batch_size, channels, height, width]
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, num_classes)
