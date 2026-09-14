from torch import nn

from trajectory_prediction.models.gru import GRUPredictor
from trajectory_prediction.models.transformer import TransformerPredictor


def build_model(config: dict) -> nn.Module:
    name = config["name"]
    kwargs = {key: value for key, value in config.items() if key != "name"}
    if name == "gru":
        return GRUPredictor(**kwargs)
    if name == "transformer":
        return TransformerPredictor(**kwargs)
    raise ValueError(f"unknown model: {name}")
