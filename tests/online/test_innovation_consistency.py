import copy

import numpy as np
import pytest
import torch
from test_horizon_gate import example_arrays

from trajectory_prediction.online.innovation_consistency import InnovationDataset, weighted_revision


def test_innovation_absolute_time_prefix_and_labels():
    a=example_arrays()
    a["cv"][0,0,1]=a["position"][0,1]+[.3,.4]
    data=InnovationDataset(a)
    assert data.innovation[0,1] == pytest.approx(.5)
    assert data.revision_weights[0,1] == pytest.approx(np.exp(-1))
    np.testing.assert_array_equal(data.revision_weights[:,0],0)
    assert data.train_mean_weight == pytest.approx(data.revision_weights[:,1:].mean(dtype=np.float64))
    b=copy.deepcopy(a)
    b["target"][:]=1e6
    b["position"][:,5:]+=100
    b["cv"][:,5:]+=300
    other=InnovationDataset(b)
    np.testing.assert_array_equal(data.revision_weights[:,:5],other.revision_weights[:,:5])
    order=[2,0,1]
    reordered=InnovationDataset({k:v[order] if k not in ("map","valid","age_s","quality") else v[:,order] for k,v in a.items()})
    np.testing.assert_array_equal(data.revision_weights[order],reordered.revision_weights)


def test_weighted_revision_update_mean_not_weight_normalized_and_gradients():
    now=torch.zeros((2,12,2),requires_grad=True)
    past=torch.zeros_like(now,requires_grad=True)
    shift=torch.tensor([2.,4.])[:,None,None]*torch.tensor([1.,0.])[None,None]
    mask=torch.arange(12)[None].expand(2,-1)>=torch.tensor([1,10])[:,None]
    loss=weighted_revision(now+shift,past,mask,torch.ones(2,dtype=torch.bool),torch.tensor([1.,0.]))
    assert loss.item()==pytest.approx(1.)
    loss.backward()
    torch.testing.assert_close(now.grad,-past.grad)
    assert now.grad[0].abs().sum()>0 and now.grad[1].abs().sum()==0


def test_first_update_and_zero_weight_are_connected_finite_zero():
    x=torch.ones((2,12,2),requires_grad=True)
    y=torch.zeros_like(x,requires_grad=True)
    for previous,weight in [(False,1.),(True,0.)]:
        loss=weighted_revision(x,y,torch.ones((2,12),dtype=torch.bool),torch.full((2,),previous),torch.full((2,),weight))
        assert loss.item()==0 and torch.isfinite(loss)
        assert all(g is not None for g in torch.autograd.grad(loss,(x,y),retain_graph=True))
    with pytest.raises(ValueError,match="weight"):
        weighted_revision(x,y,torch.ones((2,12),dtype=torch.bool),torch.ones(2,dtype=torch.bool),torch.tensor([float("nan"),1.]))


def test_weight_fingerprint_binds_observed_innovation_even_when_head_features_same():
    a=example_arrays()
    d=InnovationDataset(a)
    before=d.fingerprint()
    d.revision_weights[0,1]=.123
    assert d.fingerprint()!=before
