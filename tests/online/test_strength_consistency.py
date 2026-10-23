import pytest
import torch

from trajectory_prediction.online.innovation_consistency import weighted_revision
from trajectory_prediction.online.strength_consistency import ARMS, strength_settings


def test_strength_matrix_mean_matching_and_invalid_inputs():
    assert ARMS == ("fixed_005", "adaptive_matched_005", "fixed_010", "adaptive_matched_010")
    for arm in ARMS:
        s=strength_settings(arm,.4)
        alpha=.05 if arm.endswith("005") else .1
        assert s["target_mean_weight"]==alpha
        assert s["revision_kind"]==("revision_adaptive" if arm.startswith("adaptive") else "revision_fixed")
        assert s["revision_weight"]*(.4 if arm.startswith("adaptive") else 1)==pytest.approx(alpha)
    for mu in (0.,-.1,1.1,float("nan"),float("inf")):
        with pytest.raises(ValueError,match="mean"):
            strength_settings(ARMS[0],mu)
    with pytest.raises(ValueError,match="arm"):
        strength_settings("unknown",.5)


def test_constant_mean_control_loss_and_both_gradients_equal_fixed():
    now=torch.randn(3,12,2,requires_grad=True)
    prev=torch.randn(3,12,2,requires_grad=True)
    mask=torch.arange(12)[None,:]>=torch.tensor([1,4,8])[:,None]
    pairs=torch.tensor([False,True,True]);mu=.4;alpha=.1
    fixed=alpha*weighted_revision(now,prev,mask,pairs,torch.ones(3))
    control=(alpha/mu)*weighted_revision(now,prev,mask,pairs,torch.full((3,),mu))
    torch.testing.assert_close(fixed,control)
    a=torch.autograd.grad(fixed,(now,prev),retain_graph=True)
    b=torch.autograd.grad(control,(now,prev))
    for x,y in zip(a,b,strict=True):
        torch.testing.assert_close(x,y)
        assert x.abs().sum()>0
