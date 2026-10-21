import pytest
import torch
from pydantic import ValidationError

from src.components.model import MobileNetV3Size, build_mobilenet_v3
from src.config import AppConfig, ModelConfig


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


@pytest.mark.parametrize(
    ("size", "expected_in_features"),
    [("small", 1024), ("large", 1280)],
)
def test_build_mobilenet_v3_size_variants(
    size: MobileNetV3Size, expected_in_features: int
) -> None:
    model = build_mobilenet_v3(num_classes=3, freeze_base=True, size=size)

    last_layer = model.classifier[-1]
    assert last_layer.in_features == expected_in_features
    assert last_layer.out_features == 3


def test_build_mobilenet_v3_default_size_is_small() -> None:
    default_model = build_mobilenet_v3(num_classes=3)
    small_model = build_mobilenet_v3(num_classes=3, size="small")

    assert sum(p.numel() for p in default_model.parameters()) == sum(
        p.numel() for p in small_model.parameters()
    )


def test_build_mobilenet_v3_invalid_size() -> None:
    with pytest.raises(ValueError, match="Unsupported MobileNetV3 size"):
        build_mobilenet_v3(num_classes=3, size="medium")  # type: ignore[arg-type]


def test_model_config_size_validation() -> None:
    assert ModelConfig().size == "small"
    assert ModelConfig(size="large").size == "large"
    with pytest.raises(ValidationError):
        ModelConfig(size="medium")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("config_path", "expected_size"),
    [
        ("configs/smoke.yaml", "small"),
        ("configs/config.yaml", "large"),
    ],
)
def test_yaml_configs_select_expected_model_size(
    config_path: str, expected_size: str
) -> None:
    assert AppConfig.from_yaml(config_path).model.size == expected_size
