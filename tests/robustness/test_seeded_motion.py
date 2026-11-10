import numpy as np
import pytest
import torch

from trajectory_prediction.robustness.motion_learning import MotionGate, fit_motion, predict_motion


def test_model_seed_reproducible_and_distinct():
    a,b,c=MotionGate(seed=2028),MotionGate(seed=2028),MotionGate(seed=2029)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert any(not torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items())
    rng=np.random.default_rng(8)
    x=rng.normal(size=(2,11,18)).astype(np.float32)
    p=rng.normal(size=(2,11,10,12,2)).astype(np.float32)
    y=rng.normal(size=(2,12,2)).astype(np.float32)
    m,s,_=fit_motion(x,p,y,'plus_ca',epochs=1,seed=2028)
    n,t,_=fit_motion(x,p,y,'plus_ca',epochs=1,seed=2028)
    assert s==t and all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items())


def test_guard_masks_ca_before_softmax_and_requires_boolean_trace():
    x=np.ones((2,11,18),np.float32)
    p=np.ones((2,11,10,12,2),np.float32)
    y=np.zeros((2,12,2),np.float32)
    mask=np.zeros((2,11),bool)
    model,stats,_=fit_motion(x,p,y,'guarded_ca',epochs=1,ca_allowed=mask)
    before,w=predict_motion(model,stats,x,p,'guarded_ca',ca_allowed=mask)
    assert not w[:,:,7:].any()
    p[:,:,7:]=1e6
    after,_=predict_motion(model,stats,x,p,'guarded_ca',ca_allowed=mask)
    np.testing.assert_array_equal(before,after)
    np.testing.assert_allclose(w.sum(-1),1,atol=1e-6)
    with pytest.raises(ValueError,match='guard'):
        fit_motion(x,p,y,'guarded_ca',epochs=1)
    ordinary,_,_=fit_motion(x,p,y,'plus_ca',epochs=1)
    unrestricted,_,_=fit_motion(x,p,y,'guarded_ca',epochs=1,ca_allowed=np.ones((2,11),bool))
    assert all(torch.equal(v,unrestricted.state_dict()[k]) for k,v in ordinary.state_dict().items())
