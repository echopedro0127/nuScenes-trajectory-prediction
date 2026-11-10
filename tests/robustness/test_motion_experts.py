import numpy as np
import pytest

from trajectory_prediction.robustness.motion_experts import bounded_path, build_motion_experts
from trajectory_prediction.robustness.packets import PROFILES, Packet, make_packets


def packets(condition='clean'):
    history=np.column_stack((np.arange(-4,1),np.zeros(5))).astype(float)
    future=np.column_stack((np.arange(1,12),np.zeros(11))).astype(float)
    return make_packets(history,future,'unit',PROFILES[condition],2027)


def test_braking_integrates_to_stop_without_reversing():
    t=np.array([0.,.5,1.,2.,6.])
    out=bounded_path(np.zeros(2),np.array([2.,0.]),-2.,t)
    np.testing.assert_allclose(out[:,0],[0.,.75,1.,1.,1.])
    assert (np.diff(out[:,0])>=0).all()
    np.testing.assert_array_equal(out[:,1],0)
    np.testing.assert_allclose(bounded_path(np.ones(2),np.zeros(2),2.,t),np.ones((5,2)))
    fast=bounded_path(np.zeros(2),np.array([2.,0.]),2.,t)
    np.testing.assert_allclose(fast[:,0],2*t+t*t)


def test_motion_experts_are_causal_and_stationary_candidate_is_constant():
    original=packets('noise_outlier')
    changed=tuple(Packet(p.source_step,p.available_step,p.position+999) if p.source_step>5 else p for p in original)
    a,b=build_motion_experts(original),build_motion_experts(changed)
    assert a.shape==(11,3,12,2) and np.isfinite(a).all()
    np.testing.assert_array_equal(a[:5],b[:5])
    np.testing.assert_array_equal(a[:,2],np.repeat(a[:,2,:1],12,axis=1))
    with pytest.raises(ValueError,match='regular'):
        build_motion_experts(packets('noise_dropout'))
