import json

import pytest

from trajectory_prediction.evaluation import scenario_synthesis as synthesis
from trajectory_prediction.evaluation.scenario_event_report import build_event_report
from trajectory_prediction.evaluation.scenario_preflight import run_preflight
from trajectory_prediction.evaluation.scenario_publication import publish_predictions
from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
from trajectory_prediction.evaluation.scenario_report import build_report
from trajectory_prediction.utils.provenance import sha256_file


def seed_rows():
    registry = [{'strategy': f's{s}_baseline7', 'train_seed': s, 'variant': 'baseline7'}
                for s in (2027, 2028, 2029)] + [{'strategy': 'raw_cv', 'train_seed': None}]
    rows = [{'role': 'independent_select', 'condition': 'clean', 'log': None,
             'group_namespace': 'all', 'tag': 'all', 'strategy': r['strategy'],
             'train_seed': r['train_seed'], 'm1_m5': {'2': {'mean_error_m': float(i + 1)}}}
            for i, r in enumerate(registry)]
    return registry, rows


def test_seed_statistics_ddof1_and_physical_not_replicated():
    registry, rows = seed_rows()
    result = synthesis.aggregate_seeds(rows, registry)
    neural = next(r for r in result if r['family'] == 'baseline7')
    metric = neural['metrics']['m1_m5.2.mean_error_m']
    assert metric['mean'] == 2
    assert metric['sample_std_ddof1'] == 1
    assert metric['valid_model_count'] == 3
    physical = next(r for r in result if r['family'] == 'raw_cv')
    assert len(physical['members']) == 1
    assert physical['members'][0]['train_seed'] is None
    assert physical['metrics']['m1_m5.2.mean_error_m']['sample_std_ddof1'] is None


def test_missing_duplicate_seed_or_wrong_metadata_rejected():
    registry, rows = seed_rows()
    with pytest.raises(ValueError, match='missing|complete'):
        synthesis.aggregate_seeds(rows[1:], registry)
    with pytest.raises(ValueError, match='duplicate'):
        synthesis.aggregate_seeds([*rows, rows[0]], registry)
    rows[0]['train_seed'] = 2028
    with pytest.raises(ValueError, match='seed'):
        synthesis.aggregate_seeds(rows, registry)


def test_missing_metric_is_not_silently_averaged():
    registry, rows = seed_rows()
    rows[0]['m1_m5']['2']['mean_error_m'] = None
    neural = next(r for r in synthesis.aggregate_seeds(rows, registry) if r['family'] == 'baseline7')
    metric = neural['metrics']['m1_m5.2.mean_error_m']
    assert metric['valid_model_count'] == 2
    assert metric['mean'] is None
    assert metric['status'] == 'not_evaluable'


def test_case_union_is_shared_and_order_independent():
    windows = [{'role': 'independent_fit', 'instance': w, 'sample': 's', 'labels': ['stop_event']}
               for w in ('a', 'b')]
    errors = [{'role': 'independent_fit', 'condition': 'noise_outlier', 'strategy': model,
               'instance': w, 'sample': 's', 'mean_error_m': value}
              for model, w, value in [('m1', 'a', 10.), ('m1', 'b', 1.), ('m2', 'a', 1.), ('m2', 'b', 9.)]]
    selected = synthesis.select_cases(windows, errors)
    assert {c['instance'] for c in selected} == {'a', 'b'}
    assert selected == synthesis.select_cases(list(reversed(windows)), list(reversed(errors)))
    assert all(c['noise_seed'] == 2027 for c in selected)


def test_end_to_end_synthesis_and_tampering(scenario_parent, tmp_path, monkeypatch):
    ready, published, queries, events, output = (tmp_path / n for n in ('ready', 'p', 'q', 'e', 's'))
    run_preflight(scenario_parent['parent'], scenario_parent['owner'], PROTOCOL_V1, ready)
    publish_predictions(ready, published)
    build_report(published, queries)
    build_event_report(queries, events)
    summary = synthesis.build_synthesis(events, output)
    assert summary['case_count'] == 6
    assert summary['formal_full_run'] is False
    assert synthesis.verify_synthesis(output)['semantic_replay_verified']
    assert len(list((output / 'cases').glob('*.png'))) == 6
    with pytest.raises(FileExistsError):
        synthesis.build_synthesis(events, output)
    image_path = next((output / 'cases').glob('*.png'))
    original_image = image_path.read_bytes()
    marker_path = output / 'completed.json'
    original_marker = marker_path.read_bytes()
    image_path.write_bytes(b'changed image')
    marker = json.loads(original_marker)
    marker['sha256'][image_path.relative_to(output).as_posix()] = sha256_file(image_path)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='rendering semantic'):
        synthesis.verify_synthesis(output)
    image_path.write_bytes(original_image)
    marker_path.write_bytes(original_marker)
    stats = json.loads((output / 'query_seed_stats.json').read_text())
    neural = next(s for s in stats if s['family'] == 'baseline7' and s['metrics'])
    next(iter(neural['metrics'].values()))['mean'] = 999.
    (output / 'query_seed_stats.json').write_text(json.dumps(stats))
    marker = json.loads((output / 'completed.json').read_text())
    marker['sha256']['query_seed_stats.json'] = sha256_file(output / 'query_seed_stats.json')
    (output / 'completed.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='semantic'):
        synthesis.verify_synthesis(output)
    with monkeypatch.context() as m:
        def interrupted(*args):
            raise RuntimeError('render interrupted')
        m.setattr(synthesis, '_plot', interrupted)
        partial = tmp_path / 'partial'
        with pytest.raises(RuntimeError, match='interrupted'):
            synthesis.build_synthesis(events, partial)
        assert not (partial / 'completed.json').exists()
        with pytest.raises(ValueError, match='incomplete'):
            synthesis.verify_synthesis(partial)
