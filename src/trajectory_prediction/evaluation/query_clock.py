"""Integer-microsecond query/GT matching; no prediction or resampling."""
import itertools

import numpy as np

LEADS = (500000, 1000000, 2000000)
TOLERANCE = 50000
STATUS = ('matched', 'beyond_window_horizon', 'no_future_target',
          'ambiguous_nearest_target', 'target_outside_tolerance')
LIMIT = int(np.iinfo(np.int64).max)


def match_target(times, query, desired):
    if (not isinstance(times, list) or not times
            or any(type(t) is not int or not 0 < t <= LIMIT for t in times)
            or any(a >= b for a,b in itertools.pairwise(times))
            or type(query) is not int or type(desired) is not int
            or not 0 < query <= desired <= LIMIT):
        raise ValueError('positive ordered integer clock required')
    index = -1
    if desired > times[-1]:
        status = STATUS[1]
    else:
        candidates = [(abs(t-desired),i) for i,t in enumerate(times) if t > query]
        if not candidates:
            status = STATUS[2]
        else:
            distance = min(d for d,_ in candidates)
            tied = [i for d,i in candidates if d == distance]
            status = STATUS[3] if len(tied) > 1 else STATUS[4] if distance > TOLERANCE else STATUS[0]
            if status == STATUS[0]:
                index = tied[0]
    return {'target_index':index, 'status':status,
            'signed_offset_us':times[index]-desired if index >= 0 else None}


def window_queries(times):
    t = np.asarray(times)
    if (t.ndim != 2 or t.shape[1] != 17 or len(t) < 1 or t.dtype != np.dtype('int64')
            or (t <= 0).any() or (t > LIMIT-max(LEADS)).any() or (np.diff(t,axis=1) <= 0).any()):
        raise ValueError('nonempty strictly increasing int64 [N,17] clocks required')
    query = t[:,5:16].copy()
    desired = query[:,:,None]+np.array(LEADS,np.int64)
    future = t[:,None,None,5:]
    available = future > query[:,:,None,None]
    distance = np.where(available, np.abs(future-desired[:,:,:,None]), LIMIT)
    nearest = distance.min(axis=-1)
    index = distance.argmin(axis=-1).astype(np.int16)+5
    codes = np.zeros(desired.shape,np.uint8)
    codes[nearest > TOLERANCE] = 4
    codes[(distance == nearest[:,:,:,None]).sum(axis=-1) > 1] = 3
    codes[np.broadcast_to(~available.any(axis=-1),desired.shape)] = 2
    codes[desired > t[:,-1,None,None]] = 1
    valid = codes == 0
    target = np.take_along_axis(t[:,None,:],index.reshape(len(t),1,-1),axis=2).reshape(index.shape)
    index[~valid] = -1
    return {'query_us':query, 'desired_us':desired, 'target_index':index,
            'status_code':codes, 'valid':valid,
            'target_us':np.where(valid,target,-1),
            'signed_offset_us':np.where(valid,target-desired,0),
            'actual_lead_us':np.where(valid,target-query[:,:,None],-1)}
