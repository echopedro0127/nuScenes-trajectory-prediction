import numpy as np
import pytest
import torch
from torch import nn

from trajectory_prediction.context.maps import LaneMap, MapConfig
from trajectory_prediction.robustness.noisy_candidates import build_candidates
from trajectory_prediction.robustness.packets import PROFILES, make_packets


class SmallGRU(nn.Module):
    def forward(self, features, mask):
        return features[:, -1:, 4:] * torch.arange(1, 13)[None, :, None] * .5


class SmallOriginal(nn.Module):
    def forward(self, features, mask):
        return features[..., 2:4] * mask[..., None]


class SmallMap(nn.Module):
    def forward(self, features, mask, points, point_mask, neighbors, neighbor_mask):
        return (features[..., 2:4] + points[:, :1, :1, :2].reshape(-1, 1, 2) * .01) * mask[..., None]


def stream(condition='clean', shift=0.):
    history=np.column_stack((np.arange(-4, 1), np.zeros(5))).astype(float) + shift
    future=np.column_stack((np.arange(1, 12), np.zeros(11))).astype(float) + shift
    return make_packets(history, future, 'unit', PROFILES[condition], 2027)


def build(packets, lanes=None):
    return build_candidates(packets, SmallGRU(), SmallOriginal(), SmallMap(),
                            lanes or LaneMap({}, MapConfig()), np.zeros(2), np.pi / 2, 10000000)


def test_noisy_initial_history_recomputed_and_map_fallback():
    clean, noisy=build(stream()), build(stream('noise_05'))
    assert clean['candidates'].shape == (11, 7, 12, 2)
    assert noisy['features'].shape == (11, 18)
    assert np.isfinite(noisy['features']).all()
    assert not np.array_equal(clean['base'], noisy['base'])
    np.testing.assert_array_equal(noisy['candidates'][:, 5], noisy['candidates'][:, 6])
    assert not noisy['map_coverage'].any()


def test_future_packets_cannot_change_published_prefix():
    from trajectory_prediction.robustness.packets import Packet
    packets=stream('noise_outlier')
    poisoned=tuple(Packet(p.source_step, p.available_step, p.position + 999) if p.source_step>5 else p for p in packets)
    a,b=build(packets),build(poisoned)
    for name in ('candidates','features','map_coverage'):
        np.testing.assert_array_equal(a[name][:5], b[name][:5])


def test_translation_does_not_expose_true_origin_to_network():
    a,b=build(stream()),build(stream(shift=17.))
    np.testing.assert_allclose(b['candidates']-17, a['candidates'], atol=1e-5)
    np.testing.assert_allclose(a['features'], b['features'], atol=1e-5)


def test_map_query_uses_each_arrived_measurement():
    class RecordingMap:
        def __init__(self):
            self.positions=[]
        def query(self, observation, xy, yaw):
            self.positions.append(observation.observed_position.copy())
            return LaneMap({}, MapConfig()).query(observation, xy, yaw)
    lanes=RecordingMap()
    packets=stream('noise_05')
    build(packets,lanes)
    assert len(lanes.positions)==11
    np.testing.assert_allclose(lanes.positions, [p.position for p in packets if p.source_step>0])


def test_delayed_packets_rejected_until_separate_protocol():
    with pytest.raises(ValueError,match='regular'):
        build(stream('noise_delay_500ms'))


def test_nonempty_map_coordinates_follow_noisy_origin_translation():
    poses=np.column_stack((np.arange(30),np.zeros(30),np.zeros(30)))
    first=LaneMap({'lane':poses},MapConfig())
    shifted=poses.copy()
    shifted[:,:2]+=17
    second=LaneMap({'lane':shifted},MapConfig())
    a,b=build(stream('noise_05'),first),build(stream('noise_05',17),second)
    assert a['map_coverage'].any()
    assert not np.array_equal(a['candidates'][:,5],a['candidates'][:,6])
    np.testing.assert_allclose(a['candidates'],b['candidates']-17,atol=2e-5)
