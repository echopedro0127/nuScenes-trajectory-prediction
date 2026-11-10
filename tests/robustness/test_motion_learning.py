import numpy as np
import torch

from trajectory_prediction.robustness.motion_learning import fit_motion, predict_motion
from trajectory_prediction.robustness.reliability_learning import fit_gate


def test_baseline_exactly_reproduces_old_training_and_masks_extra_heads():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(3,11,18)).astype(np.float32)
    c=rng.normal(size=(3,11,10,12,2)).astype(np.float32)
    y=rng.normal(size=(3,12,2)).astype(np.float32)
    old,stats,log=fit_gate(x,c[:,:,:7].copy(),y,'innovation',epochs=2)
    new,newstats,newlog=fit_motion(x,c,y,'baseline7',epochs=2)
    assert sum(p.numel() for p in new.parameters())==938
    assert stats==newstats and log==newlog
    for k,v in old.state_dict().items():
        assert torch.equal(v,new.legacy.state_dict()[k])
    before,w=predict_motion(new,stats,x,c,'baseline7')
    assert not w[:,:,7:].any()
    c[:,:,7:]=1e6
    after,_=predict_motion(new,stats,x,c,'baseline7')
    np.testing.assert_array_equal(before,after)


def test_stopping_expert_can_receive_weight_and_learn():
    x=np.zeros((96,11,18),np.float32)
    c=np.full((96,11,10,12,2),10,np.float32)
    c[:,:,9]=0
    y=np.zeros((96,12,2),np.float32)
    model,stats,_=fit_motion(x,c,y,'plus_stop',epochs=20,batch_size=128)
    forecast,w=predict_motion(model,stats,x,c,'plus_stop')
    assert not w[:,:,7].any()
    assert float(w[:,:,9].mean())>.3
    assert forecast.mean()<6
