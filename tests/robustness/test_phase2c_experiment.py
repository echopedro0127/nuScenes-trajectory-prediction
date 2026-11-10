import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.robustness.phase2c_experiment import publish, verify_output


def test_ten_expert_pipeline_roundtrip_and_tamper(tmp_path):
    rng=np.random.default_rng(9)
    datasets,records={},{}
    for role in ('independent_fit','independent_select'):
        datasets[role]={'features':rng.normal(size=(2,11,18)).astype(np.float32),
                        'candidates':rng.normal(size=(2,11,10,12,2)).astype(np.float32),
                        'target':rng.normal(size=(2,12,2)).astype(np.float32)}
        records[role]=[{'role':role,'row':i,'log':role,'condition':'clean','seed':2027} for i in range(2)]
    baseline=pd.DataFrame(columns=['role','row','log','condition','seed','strategy','ade','fde','jitter'])
    thresholds={'low_speed_mps':2.,'turn_yaw_deg':15.,'high_curvature_yaw_deg':45.,'acceleration_mps2':1.}
    output=tmp_path/'run'
    summary=publish(output,datasets,records,baseline,{},thresholds,epochs=1)
    assert summary['rows']==28 and not summary['holdout_evaluated']
    assert verify_output(output)['all_metrics_groups_weights_replayed']
    with pytest.raises(FileExistsError):
        publish(output,datasets,records,baseline,{},thresholds,epochs=1)
    (output/'weights.json').write_text('[]')
    with pytest.raises(ValueError,match='checksum'):
        verify_output(output)
