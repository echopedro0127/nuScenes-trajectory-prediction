import json

import numpy as np
import pytest
from test_physics_clean_pipeline import physics_parent as physics_parent  # noqa: PLC0414
from test_physics_gaussian_pipeline import gaussian_parent as gaussian_parent  # noqa: PLC0414
from test_query_audit_data import audit_data as audit_data  # noqa: PLC0414
from test_query_audit_pipeline import audit_parents as audit_parents  # noqa: PLC0414

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.track_sources import sources
from trajectory_prediction.utils.provenance import sha256_file


@pytest.fixture
def outlier_parent(gaussian_parent,tmp_path,monkeypatch):
    from trajectory_prediction.evaluation.physics_gaussian import prepare_gaussian
    out=tmp_path/'gaussian';prepare_gaussian(*gaussian_parent,out)
    owner=tmp_path/'gaussian-owner';(owner/'scripts').mkdir(parents=True)
    (owner/'scripts/run_realclock_physics_gaussian.py').write_text(
        "import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--output-root');p.add_argument('--verify-only',action='store_true');a=p.parse_args();r=Path(a.output_root);s=json.loads((r/'summary.json').read_text());i=json.loads((r/'run.json').read_text())['identity'];print(json.dumps({**s,'identity':i,'gaussian_semantically_replayed':True,'saved_predictions_independently_rescored':True,'scalar_reference_verified':True}))\n")
    run=json.loads((out/'run.json').read_text());run['binding']['sources']=sources(owner);run['identity']=digest(run['binding'])
    (out/'run.json').write_text(json.dumps(run))
    marker=json.loads((out/'completed.json').read_text());marker['identity']=run['identity'];marker['sha256']['run.json']=sha256_file(out/'run.json')
    (out/'completed.json').write_text(json.dumps(marker))
    # Only the synthetic fixture substitutes an identity; production has no CLI override.
    from trajectory_prediction.evaluation import outlier_sources
    monkeypatch.setattr(outlier_sources,'APPROVED_PARENT_IDENTITY',run['identity'],raising=False)
    return out,owner


def test_28_combinations_exact_parents_and_verify(outlier_parent,tmp_path):
    from trajectory_prediction.evaluation.physics_outlier import prepare_outlier, verify_outlier
    out=tmp_path/'outlier';s=prepare_outlier(*outlier_parent,out)
    assert s['combinations']==28 and s['valid_pairs_per_combination']==58
    assert len(s['seed_results'])==28 and len(s['seed_macro_results'])==8
    assert len(s['regressions'])==8
    assert verify_outlier(out)['saved_predictions_independently_rescored'] is True
    with pytest.raises(FileExistsError):prepare_outlier(*outlier_parent,out)


@pytest.mark.parametrize('kind',['outlier','diagnostic','prediction','aggregate','proof','summary','extra'])
def test_rehashed_tamper_rejected(outlier_parent,tmp_path,kind):
    from trajectory_prediction.evaluation.physics_outlier import prepare_outlier, verify_outlier
    out=tmp_path/'outlier';prepare_outlier(*outlier_parent,out)
    names={'outlier':'source_noise/2027.npz','diagnostic':'gaussian_outlier/seed_2027/kf_nis_actual_dt.npz',
           'prediction':'gaussian_outlier/seed_2027/cv_actual_dt.npz','aggregate':'aggregates/gaussian_outlier/kf_actual_dt.npz',
           'proof':'parent_proof.json','summary':'summary.json','extra':'extra.json'}
    path=out/names[kind]
    if path.suffix=='.npz':
        with np.load(path) as f:data={k:f[k] for k in f.files}
        key={'outlier':'outliers','diagnostic':'nis','prediction':'predictions','aggregate':'mean_error_over_noise_seeds'}[kind]
        data[key].flat[0]+=1;np.savez_compressed(path,**data)
    else:
        data=json.loads(path.read_text()) if path.exists() else {};data['forged']=True;path.write_text(json.dumps(data))
    marker=json.loads((out/'completed.json').read_text());marker['sha256'][names[kind]]=sha256_file(path)
    (out/'completed.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError):verify_outlier(out)


def test_rescore_rebuilds_aggregate_and_diagnostics(outlier_parent,tmp_path):
    from trajectory_prediction.evaluation import physics_outlier as mod
    out=tmp_path/'outlier';mod.prepare_outlier(*outlier_parent,out)
    info=mod.snapshot(*outlier_parent,json.loads((out/'parent_proof.json').read_text()))
    summary=json.loads((out/'summary.json').read_text())
    summary['seed_macro_results'][0]['groups'][0]['mean_window_error_m']+=1
    with pytest.raises(ValueError,match='rescore'):mod.rescore(out,info,summary)
    summary=json.loads((out/'summary.json').read_text());summary['diagnostics'][0]['groups'][0]['tp']+=1
    with pytest.raises(ValueError,match='rescore'):mod.rescore(out,info,summary)


def test_source_change_and_interruption(outlier_parent,tmp_path,monkeypatch):
    from trajectory_prediction.evaluation import physics_outlier as mod
    def fail(*args):raise RuntimeError('interrupted')
    with monkeypatch.context() as m:
        m.setattr(mod,'_write_bytes',fail)
        with pytest.raises(RuntimeError):mod.prepare_outlier(*outlier_parent,tmp_path/'partial')
    assert not (tmp_path/'partial/completed.json').exists()
    (outlier_parent[1]/'scripts/new.py').write_text('changed=True')
    with pytest.raises(ValueError):mod.prepare_outlier(*outlier_parent,tmp_path/'bad-source')


def test_self_consistent_alternate_parent_rejected(outlier_parent):
    from trajectory_prediction.evaluation.outlier_sources import snapshot
    parent,owner=outlier_parent
    run=json.loads((parent/'run.json').read_text());run['binding']['substituted_parent']=True
    run['identity']=digest(run['binding']);(parent/'run.json').write_text(json.dumps(run))
    marker=json.loads((parent/'completed.json').read_text());marker['identity']=run['identity']
    marker['sha256']['run.json']=sha256_file(parent/'run.json');(parent/'completed.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError,match='approved Gaussian identity'):snapshot(parent,owner)
