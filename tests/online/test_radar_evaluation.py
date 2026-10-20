import numpy as np
import torch
from test_radar_corrector import data_fixture

from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.radar_corrector import RadarDataset, clone_fallback
from trajectory_prediction.online.radar_evaluation import evaluate_radar, recovery_table


def test_six_strategies_keep_all_condition_updates_and_fallback_parity():
    torch.set_num_threads(1)
    arrays, cache, base = data_fixture()
    cache["filtered_position"], cache["filtered_velocity"] = cache["position"].copy(), cache["velocity"].copy()
    cache["present"][:] = False
    original = ResidualCorrector(hidden_dim=8)
    models = torch.nn.ModuleDict({"original": original, "radar": clone_fallback(original, 8), "matched_radar": ResidualCorrector(hidden_dim=8)})
    # clone_fallback deep copies, preserving strict original values.
    values, times = evaluate_radar(RadarDataset(arrays, cache, base), models, batch_size=16)
    assert set(values) == {"frozen", "sliding", "filtered", "original", "radar", "matched_radar"}
    for result in values.values():
        assert result.shape == (5, 3, 11, 3) and np.isfinite(result).all()
        assert not result[:, :, 0, 2].any()
    np.testing.assert_array_equal(values["radar"], values["original"])
    np.testing.assert_array_equal(values["frozen"][0], values["frozen"][4])
    assert all(seconds >= 0 for seconds in times.values())


def test_recovery_requires_two_steps_and_preserves_censoring():
    arrays, _, _ = data_fixture()
    metrics = np.zeros((5, 3, 11, 3))
    metrics[4, 0, 6, 0] = 1
    metrics[4, 0, 7:, 0] = 0
    metrics[4, 1, 6:, 0] = 1
    metrics[4, 1, 10, 0] = 0  # last point alone does not prove recovery
    table = recovery_table({"radar": metrics}, arrays, 2027)
    assert table.iloc[0].recovery_seconds == .5
    assert not table.iloc[1].recovered and np.isnan(table.iloc[1].recovery_seconds)
    assert table.iloc[1].censor_seconds == 2
    assert table.iloc[2].recovery_seconds == 0
