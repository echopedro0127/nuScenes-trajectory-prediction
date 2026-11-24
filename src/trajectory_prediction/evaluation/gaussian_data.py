"""Noisy measurement origin with pristine GT retained in the same translated frame."""
import numpy as np


def noisy_frame(base,normals,sigma):
    z = np.asarray(normals);xy = base['positions_local'];origin = base['origin_xy']
    if (type(sigma) not in (int,float) or sigma not in (0.,.5,1.)
            or z.shape != xy.shape or z.ndim != 3 or z.shape[1:] != (17,2)
            or z.dtype != np.dtype('float64') or xy.dtype != np.dtype('float64')
            or origin.shape != (len(z),2) or not np.isfinite(z).all()
            or not np.isfinite(xy).all() or not np.isfinite(origin).all()):
        raise ValueError('finite float64 source normals and frozen sigma required')
    try:
        with np.errstate(over='raise',invalid='raise'):
            offset = sigma*z[:,0]
            observed = xy[:,:16]+sigma*(z[:,:16]-z[:,:1])
            truth = xy-offset[:,None,:]
            noisy_origin = origin+offset
    except FloatingPointError as error:raise ValueError('noisy frame overflow') from error
    if not all(np.isfinite(a).all() for a in (offset,observed,truth,noisy_origin)):
        raise ValueError('nonfinite noisy frame')
    evaluation = {**base,'positions_local':truth,'origin_xy':noisy_origin}
    context = {'observations_local':observed,'origin_offset_xy':offset,'origin_xy':noisy_origin}
    return observed,evaluation,context
