import numpy as np
import pandas as pd

from trajectory_prediction.robustness.maneuver_diagnostics import build_tags, grouped_metrics

THRESHOLDS={'low_speed_mps':2.,'turn_yaw_deg':15.,'high_curvature_yaw_deg':45.,'acceleration_mps2':1.}


def test_overlapping_tags_count_unique_trajectories_not_noise_seeds():
    target=np.column_stack((np.arange(1,13)*.25,np.zeros(12))).astype(np.float32)
    records=[{'role':'independent_select','row':8,'log':'log1','condition':'clean','seed':seed}
             for seed in (2027,2028,2029)]
    tags=build_tags({'independent_select':{'target':np.repeat(target[None],3,0)}},
                    {'independent_select':records},THRESHOLDS)
    assert tags[0]['tags']==['low_speed','straight'] and len(tags)==1
    frame=pd.DataFrame([{**r,'strategy':'cv','ade':2.,'fde':3.,'jitter':1.} for r in records])
    grouped=grouped_metrics(frame,tags)
    assert {g['tag'] for g in grouped}=={'low_speed','straight'}
    assert all(g['unique_trajectories']==1 and g['logs']==1 and g['trajectory_seed_records']==3 for g in grouped)


def test_future_tags_do_not_modify_causal_feature_arrays():
    from trajectory_prediction.robustness.diagnostic_controls import transform_features
    x=np.ones((1,11,18),np.float32)
    before=transform_features(x,'time_innovation')
    records={'independent_fit':[{'role':'independent_fit','row':0}]}
    for future in (np.zeros((1,12,2),np.float32),np.ones((1,12,2),np.float32)*100):
        build_tags({'independent_fit':{'target':future}},records,THRESHOLDS)
        np.testing.assert_array_equal(before,transform_features(x,'time_innovation'))
