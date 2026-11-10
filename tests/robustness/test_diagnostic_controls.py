import numpy as np
import torch

from trajectory_prediction.robustness.diagnostic_controls import (
    covariance_lookup,
    expert_mask,
    fit_control,
    predict_control,
    transform_features,
)
from trajectory_prediction.robustness.reliability_learning import fit_gate


def toy():
    rng=np.random.default_rng(4)
    return (rng.normal(size=(3,11,18)).astype(np.float32),
            rng.normal(size=(3,11,7,12,2)).astype(np.float32),
            rng.normal(size=(3,12,2)).astype(np.float32))


def test_time_features_do_not_read_covariance_or_future_updates():
    x,_,_=toy()
    changed=x.copy()
    changed[:,:,6:12]=999
    changed[:,5:,:6]+=999
    a,b=transform_features(x,'motion_time'),transform_features(changed,'motion_time')
    np.testing.assert_array_equal(a[:,:5],b[:,:5])
    np.testing.assert_array_equal(a[:,:,12:],np.zeros_like(a[:,:,12:]))
    np.testing.assert_allclose(a[0,:,6],np.arange(1,12)/11)
    np.testing.assert_array_equal(x,transform_features(x,'full'))


def test_covariance_can_be_reconstructed_from_step_without_observations():
    from test_noisy_candidates import build, stream
    for condition in ('clean','noise_05','noise_outlier'):
        np.testing.assert_array_equal(covariance_lookup(),build(stream(condition))['features'][:,6:12])


def test_full_control_exactly_reproduces_legacy_training():
    x,c,y=toy()
    old,stats,log=fit_gate(x,c,y,'innovation',epochs=2)
    new,newstats,newlog=fit_control(x,c,y,'full',epochs=2)
    assert stats==newstats and log==newlog
    for k,v in old.state_dict().items():
        assert torch.equal(v,new.state_dict()[k])


def test_disabled_experts_have_zero_weight_and_no_prediction_effect():
    x,c,y=toy()
    for variant,count in [('physical_candidates',4),('no_map_candidate',6)]:
        assert expert_mask(variant).sum()==count
        model,stats,_=fit_control(x,c,y,variant,epochs=1)
        before,w=predict_control(model,stats,x,c,variant)
        assert not w[:,:,count:].any()
        changed=c.copy()
        changed[:,:,count:]=1e6
        after,_=predict_control(model,stats,x,changed,variant)
        np.testing.assert_array_equal(before,after)
        np.testing.assert_allclose(w.sum(-1),1,atol=1e-6)
