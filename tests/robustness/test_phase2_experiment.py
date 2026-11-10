import json

import numpy as np
import pytest

from trajectory_prediction.robustness.phase2_experiment import publish_learning, verify_learning


def test_small_learning_pipeline_round_trip_and_tamper(tmp_path):
    rng=np.random.default_rng(7)
    datasets,records={},{}
    for role in ('independent_fit','independent_select'):
        datasets[role]={'features':rng.normal(size=(2,11,18)).astype(np.float32),
                        'candidates':rng.normal(size=(2,11,7,12,2)).astype(np.float32),
                        'target':rng.normal(size=(2,12,2)).astype(np.float32)}
        records[role]=[{'role':role,'log':role,'condition':'clean','seed':2027,'row':i} for i in range(2)]
    output=tmp_path/'run'
    summary=publish_learning(output,datasets,records,{'references':{},'purpose':'unit'},epochs=2)
    assert not summary['holdout_evaluated'] and summary['rows']==48
    assert verify_learning(output)['all_metrics_replayed']
    with pytest.raises(FileExistsError):
        publish_learning(output,datasets,records,{},epochs=2)
    marker=json.loads((output/'completed.json').read_text())
    assert 'motion.pt' in marker['sha256']
    (output/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        verify_learning(output)


def test_frozen_models_accept_config_path_round_tripped_through_json(tmp_path,monkeypatch):
    import torch
    from torch import nn

    from trajectory_prediction.config import load_yaml
    from trajectory_prediction.evaluation.runner import _mapping_sha256
    from trajectory_prediction.models.factory import build_model
    from trajectory_prediction.robustness import phase2_data
    from trajectory_prediction.utils.provenance import sha256_file

    config=tmp_path/'gru.yaml'
    config.write_text('model:\n  name: gru\n  input_dim: 6\n  hidden_dim: 4\n  layers: 1\n  future_steps: 12\n')
    gru=build_model(load_yaml(config)['model'])
    (tmp_path/'gru-seed-2027').mkdir()
    (tmp_path/'map-training/seed-2027').mkdir(parents=True)
    checkpoint=tmp_path/'gru-seed-2027/best.pt'
    torch.save({'model':gru.state_dict(),'metadata':{'role':'fit'}},checkpoint)
    meta={'base_model':{'checkpoint_sha256':sha256_file(checkpoint),
                        'model_config_sha256':_mapping_sha256(load_yaml(config))},
          'upstream_training':{'role':'fit'}}
    torch.save({'input_metadata':meta},tmp_path/'map-training/seed-2027/best.pt')
    monkeypatch.setattr(phase2_data,'load_pair',lambda path,metadata:(nn.ModuleDict(),{}))
    serialized=json.loads(json.dumps({'config':str(config)}))
    model,_,_=phase2_data.load_models(tmp_path,serialized['config'])
    assert not model.training
