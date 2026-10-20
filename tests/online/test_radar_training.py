import pytest
import torch
from test_radar_corrector import data_fixture

from trajectory_prediction.online.radar_corrector import RadarDataset
from trajectory_prediction.online.radar_training import train_radar_suite


def test_radar_training_tracks_scene_selection_and_rejects_stale_reuse(tmp_path):
    torch.set_num_threads(1)
    bundle = data_fixture()
    train, selection = RadarDataset(*bundle, sample_indices=[0, 1]), RadarDataset(*bundle, sample_indices=[2])
    metadata = {"context_contract": {"config": {}, "base": {}}, "cache_sha256": "a" * 64}
    models = train_radar_suite(train, selection, metadata, tmp_path, epochs=2, batch_size=64, hidden_dim=8, seed=7, patience=1)
    assert set(models) == {"original", "radar", "matched_radar"}
    payload = torch.load(tmp_path / "best.pt", weights_only=True)
    assert payload["metadata"]["train_scenes"] == ["a"]
    assert payload["metadata"]["selection_scenes"] == ["b"]
    assert all(1 <= stage["best_epoch"] <= stage["completed_epochs"] <= 2 for stage in payload["metadata"]["stage_progress"].values())
    with pytest.raises(ValueError, match="overlap"):
        train_radar_suite(train, train, metadata, tmp_path / "overlap", epochs=1)
    with pytest.raises(ValueError, match="identity"):
        train_radar_suite(train, selection, metadata, tmp_path, epochs=1)
