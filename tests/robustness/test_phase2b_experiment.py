import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.robustness.phase2b_experiment import publish, verify_output

THRESHOLDS={'low_speed_mps':2.,'turn_yaw_deg':15.,'high_curvature_yaw_deg':45.,'acceleration_mps2':1.}


def test_small_diagnostic_pipeline_replays_and_rejects_tamper(tmp_path):
    rng=np.random.default_rng(1)
    datasets,records={},{}
    for role in ('independent_fit','independent_select'):
        datasets[role]={'features':rng.normal(size=(2,11,18)).astype(np.float32),
                        'candidates':rng.normal(size=(2,11,7,12,2)).astype(np.float32),
                        'target':rng.normal(size=(2,12,2)).astype(np.float32)}
        records[role]=[{'role':role,'row':i,'log':role,'condition':'clean','seed':2027} for i in range(2)]
    parent_frame=pd.DataFrame(columns=['role','row','log','condition','seed','strategy','ade','fde','jitter'])
    output=tmp_path/'run'
    summary=publish(output,datasets,records,parent_frame,{},THRESHOLDS,epochs=1)
    assert summary['rows']==20 and not summary['holdout_evaluated']
    assert verify_output(output)['all_metrics_and_groups_replayed']
    with pytest.raises(FileExistsError):
        publish(output,datasets,records,parent_frame,{},THRESHOLDS,epochs=1)
    (output/'events.json').write_text('[]')
    with pytest.raises(ValueError,match='checksum'):
        verify_output(output)
