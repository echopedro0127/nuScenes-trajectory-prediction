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
