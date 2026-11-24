"""Synthetic measurement spikes with unchanged world-coordinate truth."""
import numpy as np

from trajectory_prediction.evaluation.gaussian_data import noisy_frame


def outlier_frame(base,normals,outliers,condition):
    if condition not in ('clean','gaussian_05','gaussian_outlier'):
        raise ValueError('frozen outlier condition required')
    spikes=np.asarray(outliers)
    if (spikes.shape != base['positions_local'].shape or spikes.dtype != np.dtype('float64')
            or not np.isfinite(spikes).all()):raise ValueError('finite float64 source spikes required')
    # Preserve the frozen Gaussian arithmetic exactly for parent regression.
    if condition != 'gaussian_outlier':return noisy_frame(base,normals,0. if condition=='clean' else .5)
    observed,evaluation,context=noisy_frame(base,normals,.5)
    with np.errstate(over='raise',invalid='raise'):
        offset=spikes[:,0]
        observed=observed+(spikes[:,:16]-offset[:,None])
        evaluation={**evaluation,'positions_local':evaluation['positions_local']-offset[:,None],
                    'origin_xy':evaluation['origin_xy']+offset}
        context={**context,'observations_local':observed,'origin_offset_xy':context['origin_offset_xy']+offset,
                 'origin_xy':evaluation['origin_xy']}
    if not all(np.isfinite(a).all() for a in (observed,evaluation['positions_local'],context['origin_xy'])):
        raise ValueError('nonfinite outlier frame')
    return observed,evaluation,context
