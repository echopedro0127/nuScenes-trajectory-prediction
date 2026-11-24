import numpy as np
import pytest


def test_spiked_origin_and_pristine_gt():
    from trajectory_prediction.evaluation.outlier_data import outlier_frame
    xy=np.arange(68,dtype=np.float64).reshape(2,17,2);xy-=xy[:,:1].copy()
    base={'positions_local':xy,'origin_xy':np.array([[100.,200.],[300.,400.]])}
    z=np.ones_like(xy);o=np.zeros_like(xy);o[:,0]=[3.,4.];o[:,7]=[-3.,4.]
    obs,ev,ctx=outlier_frame(base,z,o,'gaussian_outlier')
    np.testing.assert_array_equal(obs[:,0],np.zeros((2,2)))
    np.testing.assert_array_equal(ev['positions_local']+ctx['origin_xy'][:,None],xy+base['origin_xy'][:,None])
    np.testing.assert_array_equal(obs+ctx['origin_xy'][:,None],xy[:,:16]+base['origin_xy'][:,None]+.5*z[:,:16]+o[:,:16])
    changed=o.copy();changed[:,16]=1e6
    np.testing.assert_array_equal(obs,outlier_frame(base,z,changed,'gaussian_outlier')[0])
    assert obs.shape==(2,16,2)


def test_gaussian_and_clean_exact_parent():
    from trajectory_prediction.evaluation.gaussian_data import noisy_frame
    from trajectory_prediction.evaluation.outlier_data import outlier_frame
    rng=np.random.default_rng(9);xy=rng.normal(size=(3,17,2));xy-=xy[:,:1].copy()
    base={'positions_local':xy,'origin_xy':rng.normal(size=(3,2))};z=rng.normal(size=xy.shape);o=z.copy()
    for name,sigma in [('clean',0.),('gaussian_05',.5)]:
        a,b,c=outlier_frame(base,z,o,name);x,y,t=noisy_frame(base,z,sigma)
        np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b['positions_local'],y['positions_local'])
        np.testing.assert_array_equal(c['origin_xy'],t['origin_xy'])
    with pytest.raises(ValueError):outlier_frame(base,z,o,'foreign')
