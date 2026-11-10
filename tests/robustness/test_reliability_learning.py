import numpy as np
import pytest
import torch

from trajectory_prediction.robustness.reliability_learning import (
    ReliabilityGate,
    fit_gate,
    masked_features,
    predict_gate,
)


def test_ablations_have_identical_initialization_capacity_and_mask():
    x=np.linspace(-2,2,36,dtype=np.float32).reshape(2,18)
    for name, width in [('constant',0),('motion',6),('covariance',12),('innovation',18)]:
        out=masked_features(x,np.zeros(18),np.ones(18),name)
        np.testing.assert_array_equal(out[:,:width],x[:,:width])
        assert not out[:,width:].any()
    a,b=ReliabilityGate(),ReliabilityGate()
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for x,y in zip(a.parameters(),b.parameters(),strict=True):
        assert torch.equal(x,y)


def test_training_learns_separable_experts_using_fit_statistics_only():
    rng=np.random.default_rng(4)
    x=np.zeros((128,11,18),np.float32)
    x[:,:,0]=rng.choice([-1.,1.],size=(128,1))
    candidates=np.full((128,11,7,12,2),10,np.float32)
    for i in range(128):
        candidates[i,:,0 if x[i,0,0]<0 else 1]=0
    target=np.zeros((128,12,2),np.float32)
    model,stats,log=fit_gate(x,candidates,target,'motion',epochs=20,batch_size=128)
    forecast,weights=predict_gate(model,stats,x,candidates,'motion')
    assert len(log)==20
    assert forecast.mean()<2
    np.testing.assert_allclose(weights.sum(-1),1,atol=1e-6)
    assert forecast.min()>=0 and forecast.max()<=10
    np.testing.assert_allclose(stats['mean'],x.reshape(-1,18).mean(0),atol=1e-6)
    assert len(stats['mean'])==18
    with pytest.raises(ValueError,match='target'):
        fit_gate(x,candidates,target[:1],'motion',epochs=1)
