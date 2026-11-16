import numpy as np
import pytest


def config(**changes):
    return {'rule_version': 'stop_restart_realclock_v1', 'stop_speed_mps': .3,
                'restart_speed_mps': 1., 'prior_history_us': 2000000, 'stop_dwell_us': 1000000,
                'restart_dwell_us': 1000000, 'minimum_dwell_intervals': 2, 'max_interval_us': 1000000} | changes


def motion(times, xy=None, **changes):
    from trajectory_prediction.analysis.realclock_motion import build_motion
    n = len(times)
    track = {'role': 'independent_fit', 'instance': 'instance', 'scene': 'scene', 'log': 'log',
                 'category': 'vehicle.car', 'semantic_sha256': 'track-hash', 'role_row_start': 0, 'role_row_stop': n}
    segment = {'segment_id': 'segment', 'start_index': 0, 'stop_index': n,
                   'first_annotation': 'a0', 'last_annotation': f'a{n-1}'}
    arrays = {'timestamp_us': np.array(times, np.int64),
                  'world_xy': np.array(xy if xy is not None else [[i, 0.] for i in range(n)], np.float64),
                  'annotation_token': np.array([f'a{i}' for i in range(n)]),
                  'sample_token': np.array([f's{i}' for i in range(n)])}
    return build_motion(track, segment, arrays, config(**changes), 'v1.0-trainval')


def test_config_strict_validation():
    from trajectory_prediction.analysis.realclock_motion import validate_config
    assert validate_config(config()) == config()


@pytest.mark.parametrize('changes', [{'unknown': 1}, {'stop_speed_mps': True},
    {'stop_speed_mps': float('nan')}, {'restart_speed_mps': float('inf')},
    {'stop_speed_mps': 1.}, {'stop_speed_mps': -.1}, {'stop_dwell_us': True}])
def test_config_invalid(changes):
    from trajectory_prediction.analysis.realclock_motion import validate_config
    with pytest.raises(ValueError):
        validate_config(config(**changes))


def test_nonuniform_motion_and_gap():
    result = motion([1000000, 1400000, 2000000, 3100001, 3600001])
    assert len(result['intervals']) == 4
    assert [r['valid'] for r in result['intervals']] == [True, True, False, True]
    assert result['intervals'][0]['speed_mps'] == 2.5
    assert result['intervals'][2]['reason'] == 'temporal_resolution_gap'
    assert [b['interval_count'] for b in result['blocks']] == [2, 1]
    assert result['intervals'][0]['block_id'] != result['intervals'][-1]['block_id']


def test_single_point_and_inclusive_max_dt():
    assert motion([1000000])['diagnostics'][0]['reason'] == 'no_velocity_interval'
    assert motion([1000000, 2000000])['intervals'][0]['valid'] is True
    assert motion([1000000, 1500000], [[0., 0.], [0., 0.]])['intervals'][0]['speed_mps'] == 0.


@pytest.mark.parametrize('times,xy', [([1, 1], None), ([2, 1], None), ([0, 1], None),
    ([1, 2], [[0., 0.], [float('nan'), 0.]]), ([1, 2], [[0., 0.], [float('inf'), 0.]])])
def test_motion_invalid(times, xy):
    with pytest.raises(ValueError):
        motion(times, xy)


def test_motion_id_binds_config():
    assert motion([1000000, 1500000])['blocks'][0]['block_id'] != motion(
        [1000000, 1500000], stop_speed_mps=.2)['blocks'][0]['block_id']
