import numpy as np
import torch
from test_interaction_evaluator import fixture

from trajectory_prediction.online.batched_context import BatchedContext
from trajectory_prediction.online.batched_evaluation import evaluate_batches
from trajectory_prediction.online.interaction_evaluator import evaluate_interactions


def test_batched_metrics_match_real_nine_strategy_replay_with_nonempty_context():
    torch.set_num_threads(1)
    arrays, maps, neighbors, base, models = fixture()
    for mode in ("map", "neighbors", "fusion"):
        torch.nn.init.constant_(models[mode].head[-1].weight, .1)
    maps["mask"][0, :, 0, 0] = True
    maps["lane_tokens"][0, :, 0] = "lane"
    maps["features"][0, :, 0, 0] = [1, 0, 1, 0]
    neighbors["mask"][0, :, 0, -1] = True
    neighbors["instance_tokens"][0, :, 0] = "neighbor"
    neighbors["annotation_tokens"][0, :, 0, -1] = "annotation"
    neighbors["source_timestamp_us"][0, :, 0, -1] = neighbors["timestamp_us"][0] - 100_000
    neighbors["available_timestamp_us"][0, :, 0, -1] = neighbors["timestamp_us"][0] - 100_000
    neighbors["features"][0, :, 0, -1, 0] = 1
    neighbors["features"][0, :, 0, -1, 4] = 1
    reference = evaluate_interactions(arrays, maps, neighbors, base, models)
    results, timing = evaluate_batches(BatchedContext(arrays, maps, neighbors, base), models,
                                       batch_size=4, device=torch.device("cpu"))
    for strategy, metrics in results.items():
        expected = reference[reference.strategy == strategy][["ade", "fde", "jitter"]].to_numpy()
        np.testing.assert_allclose(metrics.reshape(-1, 3), expected, rtol=1e-4, atol=1e-4)
        assert metrics[0, 0, 2] == 0
        assert timing[strategy] >= 0
    assert np.count_nonzero(results["frozen"][..., 2]) == 0
