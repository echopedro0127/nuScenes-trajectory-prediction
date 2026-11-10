import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.robustness.phase2d_experiment import publish, verify_output


def test_seeded_pipeline_replays_groups_and_rejects_tamper(tmp_path):
    rng=np.random.default_rng(2)
    datasets,records={},{}
    for role in ('independent_fit','independent_select'):
        datasets[role]={'features':rng.normal(size=(2,11,18)).astype(np.float32),
                        'candidates':rng.normal(size=(2,11,10,12,2)).astype(np.float32),
                        'target':rng.normal(size=(2,12,2)).astype(np.float32),
                        'ca_allowed':np.tile(np.arange(11)%2==0,(2,1)),
                        'guard_yaw':np.ones((2,11),np.float32),'guard_min_speed':np.ones((2,11),np.float32)}
        records[role]=[{'role':role,'row':i,'log':role,'condition':'clean','seed':2027} for i in range(2)]
    baseline=pd.DataFrame(columns=['role','row','log','condition','seed','strategy','ade','fde','jitter'])
    thresholds={'low_speed_mps':2.,'turn_yaw_deg':15.,'high_curvature_yaw_deg':45.,'acceleration_mps2':1.}
    output=tmp_path/'run'
    result=publish(output,datasets,records,baseline,{},thresholds,epochs=1,seeds=(2027,2028))
    assert result['rows']==24 and not result['holdout_evaluated']
    assert verify_output(output)['all_seed_metrics_groups_weights_replayed']
    with pytest.raises(FileExistsError):
        publish(output,datasets,records,baseline,{},thresholds,epochs=1,seeds=(2027,2028))
    (output/'seed_summary.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        verify_output(output)
