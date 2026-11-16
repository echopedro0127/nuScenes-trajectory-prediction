import copy

import pytest

from trajectory_prediction.analysis.track_timeline import build_tracks


def tables():
    times = [1000000, 1400000, 1999000, 2499000]
    samples = {f's{i}': {'token': f's{i}', 'timestamp': t, 'scene_token': 'scene',
                         'prev': f's{i-1}' if i else '', 'next': f's{i+1}' if i < 3 else ''}
               for i, t in enumerate(times)}
    chosen = [0, 1, 3]
    rows = [{'token': f'a{i}', 'sample_token': f's{k}', 'instance_token': 'i',
             'prev': f'a{i-1}' if i else '', 'next': f'a{i+1}' if i < 2 else '',
             'translation': [float(k), 0., 0.], 'rotation': [1., 0., 0., 0.]}
            for i, k in enumerate(chosen)]
    kwargs = {'samples': samples, 'instances': {'i': {'token': 'i', 'category_token': 'c',
               'nbr_annotations': 3, 'first_annotation_token': 'a0', 'last_annotation_token': 'a2'}},
              'scenes': {'scene': {'token': 'scene', 'name': 'scene-name', 'log_token': 'log'}},
              'categories': {'c': 'vehicle.car'}, 'targets_by_role': {'independent_fit': {'i'}},
              'logs_by_role': {'independent_fit': {'log'}}, 'dataset_release': 'v1.0-trainval'}
    return rows, kwargs


def test_real_clock_gap_and_row_order_invariant_identity():
    rows, kwargs = tables()
    result = build_tracks(rows, **kwargs)
    assert result == build_tracks(list(reversed(rows)), **kwargs)
    assert len(result['tracks']) == 1
    assert len(result['segments']) == 2
    assert result['gaps'][0]['missing_sample_tokens'] == ['s2']
    points = result['tracks'][0]['annotations']
    assert [p['timestamp_us'] for p in points] == [1000000, 1400000, 2499000]
    assert [p['actual_dt_us'] for p in points] == [0, 400000, 0]
    assert [p['delta_valid'] for p in points] == [False, True, False]
    changed = copy.deepcopy(rows)
    changed[1]['translation'][0] += .1
    assert result['segments'][0]['segment_id'] != build_tracks(changed, **kwargs)['segments'][0]['segment_id']


@pytest.mark.parametrize('kind', ['duplicate', 'prev', 'cycle', 'count', 'last', 'timestamp',
                                  'sample_duplicate', 'cross_log', 'missing_instance',
                                  'bad_rotation', 'bad_xy', 'sample_link', 'oversized_token',
                                  'timestamp_bool', 'timestamp_overflow', 'missing_category', 'huge_rotation'])
def test_invalid_chain_or_schema_is_not_repaired(kind):
    rows, kwargs = tables()
    if kind == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif kind == 'prev': rows[1]['prev'] = 'wrong'
    elif kind == 'cycle': rows[2]['next'] = 'a0'
    elif kind == 'count': kwargs['instances']['i']['nbr_annotations'] = 4
    elif kind == 'last': kwargs['instances']['i']['last_annotation_token'] = 'a1'
    elif kind == 'timestamp': kwargs['samples']['s1']['timestamp'] = 1000000
    elif kind == 'sample_duplicate': rows[1]['sample_token'] = 's0'
    elif kind == 'cross_log': kwargs['scenes']['scene']['log_token'] = 'outside'
    elif kind == 'missing_instance': kwargs['instances'].clear()
    elif kind == 'bad_rotation': rows[0]['rotation'] = [0.] * 4
    elif kind == 'bad_xy': rows[0]['translation'][0] = float('nan')
    elif kind == 'sample_link': kwargs['samples']['s2']['prev'] = 's0'
    elif kind == 'oversized_token': rows[0]['token'] = 'a' * 33
    elif kind == 'timestamp_bool': kwargs['samples']['s1']['timestamp'] = True
    elif kind == 'timestamp_overflow': kwargs['samples']['s1']['timestamp'] = 2**63
    elif kind == 'missing_category': kwargs['categories'].clear()
    elif kind == 'huge_rotation': rows[0]['rotation'] = [1e154] * 4
    with pytest.raises(ValueError):
        build_tracks(rows, **kwargs)


def test_role_overlap_rejected():
    rows, kwargs = tables()
    kwargs['targets_by_role']['independent_select'] = {'i'}
    kwargs['logs_by_role']['independent_select'] = {'other'}
    with pytest.raises(ValueError, match='role'):
        build_tracks(rows, **kwargs)
