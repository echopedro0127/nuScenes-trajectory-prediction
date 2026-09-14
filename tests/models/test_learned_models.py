from collections.abc import Callable

import pytest
import torch
from torch import nn

from trajectory_prediction.models.factory import build_model
from trajectory_prediction.models.gru import GRUPredictor
from trajectory_prediction.models.transformer import TransformerPredictor

ModelFactory = Callable[[], nn.Module]


def _gru() -> GRUPredictor:
    return GRUPredictor(
        input_dim=6,
        hidden_dim=64,
        layers=1,
        future_steps=12,
    )


def _transformer() -> TransformerPredictor:
    return TransformerPredictor(
        input_dim=6,
        hidden_dim=128,
        heads=4,
        layers=2,
        future_steps=12,
    )


@pytest.mark.parametrize("model_factory", [_gru, _transformer])
def test_learned_predictor_output_and_gradients(model_factory: ModelFactory) -> None:
    model = model_factory()
    features = torch.randn(4, 5, 6)
    mask = torch.ones(4, 5, dtype=torch.bool)

    prediction = model(features, mask)
    loss = prediction.square().mean()
    loss.backward()

    assert prediction.shape == (4, 12, 2)
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("model_factory", [_gru, _transformer])
def test_masked_feature_values_do_not_affect_predictions(
    model_factory: ModelFactory,
) -> None:
    torch.manual_seed(2027)
    model = model_factory().eval()
    features = torch.randn(2, 5, 6)
    mask = torch.tensor(
        [
            [False, False, True, True, True],
            [False, True, True, True, True],
        ]
    )
    changed = features.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 1_000

    with torch.no_grad():
        original_prediction = model(features, mask)
        changed_prediction = model(changed, mask)

    torch.testing.assert_close(original_prediction, changed_prediction)


@pytest.mark.parametrize("model_factory", [_gru, _transformer])
def test_learned_predictors_reject_rows_without_valid_history(
    model_factory: ModelFactory,
) -> None:
    features = torch.randn(2, 5, 6)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [False, False, False, False, False],
        ]
    )

    with pytest.raises(ValueError, match="at least one valid"):
        model_factory()(features, mask)


@pytest.mark.parametrize("model_factory", [_gru, _transformer])
@pytest.mark.parametrize(
    ("features", "mask", "message"),
    [
        (torch.randn(2, 5), torch.ones(2, 5, dtype=torch.bool), "features"),
        (torch.randn(2, 5, 5), torch.ones(2, 5, dtype=torch.bool), "input_dim"),
        (torch.randn(2, 0, 6), torch.ones(2, 0, dtype=torch.bool), "H >= 1"),
        (torch.randn(2, 5, 6), torch.ones(2, 5, 1, dtype=torch.bool), "mask"),
        (torch.randn(2, 5, 6), torch.ones(2, 4, dtype=torch.bool), "match"),
        (torch.randn(2, 5, 6), torch.ones(2, 5), "boolean"),
    ],
)
def test_learned_predictors_validate_inputs(
    model_factory: ModelFactory,
    features: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        model_factory()(features, mask)


def test_transformer_rejects_history_longer_than_position_embedding() -> None:
    model = TransformerPredictor(
        input_dim=6,
        hidden_dim=128,
        heads=4,
        layers=2,
        future_steps=12,
        max_history_steps=5,
    )

    with pytest.raises(ValueError, match="max_history_steps"):
        model(
            torch.randn(2, 6, 6),
            torch.ones(2, 6, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        (
            {
                "name": "gru",
                "input_dim": 6,
                "hidden_dim": 64,
                "layers": 1,
                "future_steps": 12,
            },
            GRUPredictor,
        ),
        (
            {
                "name": "transformer",
                "input_dim": 6,
                "hidden_dim": 128,
                "heads": 4,
                "layers": 2,
                "future_steps": 12,
            },
            TransformerPredictor,
        ),
    ],
)
def test_build_model_creates_requested_predictor(
    config: dict[str, int | str],
    expected_type: type[nn.Module],
) -> None:
    model = build_model(config)

    assert isinstance(model, expected_type)


def test_build_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model({"name": "oracle"})
