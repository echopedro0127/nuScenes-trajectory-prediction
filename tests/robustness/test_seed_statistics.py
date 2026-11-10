import numpy as np
import pandas as pd

from trajectory_prediction.robustness.seed_statistics import seed_statistics


def test_statistics_pair_training_seeds_not_noise_replicates():
    rows=[]
    for seed,offset in [(2027,0.),(2028,1.),(2029,2.)]:
        for variant,value in [('baseline7',4.+offset),('plus_ca',3.+offset),('guarded_ca',3.5+offset)]:
            for noise in (2027,2028,2029):
                rows.append({'role':'independent_select','condition':'clean','variant':variant,'train_seed':seed,
                             'seed':noise,'ade':value,'fde':value*2,'jitter':value/2})
    result=seed_statistics(pd.DataFrame(rows))
    base=next(r for r in result['across_seeds'] if r['variant']=='baseline7')
    assert base['training_seeds']==[2027,2028,2029] and base['ade_mean']==5.
    assert np.isclose(base['ade_std'],1.)
    pair=next(r for r in result['paired_summary'] if r['contrast']=='plus_ca-minus-baseline7')
    assert pair['improved_seeds']==3 and pair['training_seed_count']==3 and pair['ade_delta_mean']==-1.
