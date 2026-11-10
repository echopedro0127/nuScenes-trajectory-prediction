import numpy as np

from trajectory_prediction.robustness.motion_guard import build_guard
from trajectory_prediction.robustness.packets import PROFILES, Packet, make_packets


def test_straight_motion_allowed_and_stationary_blocked():
    points=np.column_stack((np.arange(-4,12),np.zeros(16))).astype(float)
    for value,allowed in [(points,True),(np.zeros_like(points),False)]:
        p=make_packets(value[:5],value[5:],'guard',PROFILES['clean'],2027)
        assert (build_guard(p)['ca_allowed']==allowed).all()


def test_observed_turn_and_guard_prefix_are_causal():
    angle=np.arange(-4,12)*.3
    points=np.column_stack((5*np.cos(angle),5*np.sin(angle)))
    p=make_packets(points[:5],points[5:],'turn',PROFILES['clean'],2027)
    changed=tuple(Packet(q.source_step,q.available_step,q.position+999) if q.source_step>5 else q for q in p)
    a,b=build_guard(p),build_guard(changed)
    assert (~a['ca_allowed']).any()
    for key in a:
        np.testing.assert_array_equal(a[key][:5],b[key][:5])
