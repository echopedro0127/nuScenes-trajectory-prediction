"""Native frozen-owner integration using synthetic events and declarations."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_boundary_corrections import correction_data as correction_data  # noqa: PLC0414
from test_boundary_corrections import propose
from test_verdict_overlay_pipeline import verdict_parent as verdict_parent  # noqa: PLC0414

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.candidate_verdicts import csv_text, make_template
from trajectory_prediction.utils.provenance import sha256_file


@pytest.fixture
def source_data(correction_data):
    return correction_data


@pytest.fixture
def correction_parent(verdict_parent, tmp_path, request):
    parent, old_owner = verdict_parent
    data = json.loads((parent/'review_data.json').read_text(encoding='utf-8'))
    owner = tmp_path/'frozen-verdict-owner'
    project = Path(__file__).resolve().parents[2]
    shutil.copytree(project/'src', owner/'src', ignore=shutil.ignore_patterns('__pycache__'))
    (owner/'scripts').mkdir()
    for name in ('import_candidate_verdicts.py', 'make_verdict_template.py'):
        shutil.copyfile(project/'scripts'/name, owner/'scripts'/name)
    rows = make_template(json.loads((parent/'run.json').read_text())['identity'], data)
    if getattr(request, 'param', None) != 'pending':
        for r in rows:
            r.update(decision='needs_correction', reviewer='fixture-reader', reason='Synthetic verdict')
    annotations = tmp_path/'verdicts.csv'
    annotations.write_text(csv_text(rows), encoding='utf-8-sig', newline='')
    output = tmp_path/'verdict-overlay'
    env = {**os.environ, 'PYTHONPATH':str(owner/'src')}
    command = [sys.executable, str(owner/'scripts/import_candidate_verdicts.py'),
               '--parent', str(parent), '--parent-owner', str(old_owner),
               '--annotations', str(annotations), '--output-root', str(output)]
    result = subprocess.run(command, cwd=owner, env=env, capture_output=True, text=True,
                            encoding='utf-8', check=False)
    assert result.returncode == 0, result.stderr
    return output, owner, parent


def test_template_pending_import_and_native_replay(correction_parent, tmp_path):
    from trajectory_prediction.evaluation.correction_overlay import (
        import_overlay,
        make_template_pack,
        verify_pack,
    )
    parent, owner, _ = correction_parent
    template, output = tmp_path/'template', tmp_path/'corrections'
    assert make_template_pack(parent, owner, template)['eligible'] == 3
    assert verify_pack(template)['corrections_semantically_replayed'] is True
    annotations = tmp_path/'corrections.csv'
    annotations.write_bytes((template/'boundary_corrections.csv').read_bytes())
    assert import_overlay(parent, owner, annotations, output)['pending'] == 3
    assert verify_pack(output)['proposed'] == 0
    with pytest.raises(FileExistsError): import_overlay(parent, owner, annotations, output)


@pytest.mark.parametrize('correction_parent', ['pending'], indirect=True)
def test_no_eligible_candidates_publishes_explicit_empty_release(correction_parent, tmp_path):
    from trajectory_prediction.evaluation.correction_overlay import (
        import_overlay,
        make_template_pack,
        verify_pack,
    )
    parent, owner, _ = correction_parent
    template = tmp_path/'template'; summary = make_template_pack(parent, owner, template)
    assert summary['status'] == 'no_eligible_corrections'
    assert summary['eligible'] == summary['proposed'] == 0
    annotations = tmp_path/'empty.csv'; annotations.write_bytes((template/'boundary_corrections.csv').read_bytes())
    output = tmp_path/'overlay'; import_overlay(parent, owner, annotations, output)
    verified = verify_pack(output)
    assert verified['corrections_semantically_replayed'] is True
    assert verified['human_boundary_declarations_imported'] is False
    assert json.loads((output/'corrections.json').read_text()) == []


def test_synthetic_proposal_and_mutable_reading(correction_parent, tmp_path):
    from trajectory_prediction.evaluation.boundary_corrections import csv_text
    from trajectory_prediction.evaluation.correction_overlay import (
        import_overlay,
        make_template_pack,
        verify_pack,
    )
    from trajectory_prediction.evaluation.event_review_data import csv_text as review_csv
    parent, owner, review = correction_parent
    template = tmp_path/'template'; make_template_pack(parent, owner, template)
    rows = json.loads((template/'template.json').read_text())
    propose(rows, 'z', onset_token='c3', confirmation_token='c5')
    annotations = tmp_path/'corrections.csv'
    annotations.write_text(csv_text(rows), encoding='utf-8-sig', newline='')
    output = tmp_path/'corrections'; import_overlay(parent, owner, annotations, output)
    assert verify_pack(output)['proposed'] == 1
    queue = json.loads((review/'queue.json').read_text())
    for r in queue:
        r.update(review_status='pending', reviewer='', notes='')
    queue[0].update(review_status='reviewed', reviewer='fixture-reader', notes='Reading only')
    (review/'review_working.csv').write_text(review_csv(queue), encoding='utf-8-sig', newline='')
    assert verify_pack(output)['parent_reading_status_counts']['reviewed'] == 1
    (review/'review_working.csv').write_text('broken')
    with pytest.raises(ValueError): verify_pack(output)


@pytest.mark.parametrize('kind', ['summary', 'effective', 'protocol', 'extra', 'binding', 'input_copy', 'proof'])
def test_rehashed_tampering_rejected(correction_parent, tmp_path, kind):
    from trajectory_prediction.evaluation.correction_overlay import (
        import_overlay,
        make_template_pack,
        verify_pack,
    )
    parent, owner, _ = correction_parent
    template = tmp_path/'template'; make_template_pack(parent, owner, template)
    annotations = tmp_path/'corrections.csv'; annotations.write_bytes((template/'boundary_corrections.csv').read_bytes())
    output = tmp_path/'corrections'; import_overlay(parent, owner, annotations, output)
    names = {'summary': 'summary.json', 'effective': 'effective_events.json', 'protocol': 'protocol.json',
                 'extra': 'extra.json', 'binding': 'run.json', 'input_copy': 'annotations.csv', 'proof': 'parent_proof.json'}
    path = output/names[kind]
    marker = json.loads((output/'completed.json').read_text())
    if kind == 'input_copy': path.write_text('changed')
    else:
        value = json.loads(path.read_text()) if path.exists() else {}
        if kind == 'summary': value['proposed'] = 99
        elif kind == 'effective': value[0]['effective_event']['onset_us'] += 1
        elif kind == 'protocol': value['allow_new_events'] = True
        elif kind == 'binding':
            value['binding']['holdout_evaluated'] = 0
            value['identity'] = digest(value['binding']); marker['identity'] = value['identity']
        elif kind == 'proof': value['static_payload']['verdicts_semantically_replayed'] = False
        path.write_text(json.dumps(value), encoding='utf-8')
    marker['sha256'][path.name] = sha256_file(path)
    (output/'completed.json').write_text(json.dumps(marker), encoding='utf-8')
    with pytest.raises(ValueError): verify_pack(output)


def test_original_input_and_owner_changes_rejected(correction_parent, tmp_path):
    from trajectory_prediction.evaluation.correction_overlay import (
        import_overlay,
        make_template_pack,
        verify_pack,
    )
    parent, owner, _ = correction_parent
    template = tmp_path/'template'; make_template_pack(parent, owner, template)
    annotations = tmp_path/'corrections.csv'; annotations.write_bytes((template/'boundary_corrections.csv').read_bytes())
    output = tmp_path/'corrections'; import_overlay(parent, owner, annotations, output)
    raw = annotations.read_bytes(); annotations.write_bytes(raw+b'changed')
    with pytest.raises(ValueError): verify_pack(output)
    annotations.write_bytes(raw)
    script = owner/'scripts/import_candidate_verdicts.py'; script.write_text('changed')
    with pytest.raises(ValueError): verify_pack(output)


def test_interrupted_pack_has_no_completion(correction_parent, tmp_path, monkeypatch):
    from trajectory_prediction.evaluation import correction_overlay as mod
    parent, owner, _ = correction_parent
    def fail(*args): raise RuntimeError('interrupted')
    monkeypatch.setattr(mod, '_write_bytes', fail)
    output = tmp_path/'partial'
    with pytest.raises(RuntimeError): mod.make_template_pack(parent, owner, output)
    assert not (output/'completed.json').exists()


def test_parent_template_mode_rejected_before_native_run(correction_parent, tmp_path):
    from trajectory_prediction.evaluation.correction_overlay import make_template_pack
    parent, owner, _ = correction_parent
    # Even a rehashed mode substitution is not an eligible overlay parent.
    run = json.loads((parent/'run.json').read_text()); run['binding']['mode'] = 'template'
    run['identity'] = digest(run['binding'])
    (parent/'run.json').write_text(json.dumps(run))
    marker = json.loads((parent/'completed.json').read_text())
    marker['identity'] = run['identity']; marker['sha256']['run.json'] = sha256_file(parent/'run.json')
    (parent/'completed.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='overlay'): make_template_pack(parent, owner, tmp_path/'out')
