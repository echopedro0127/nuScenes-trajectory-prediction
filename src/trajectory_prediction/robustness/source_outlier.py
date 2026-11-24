"""Frozen independent source-keyed occurrence and direction channels."""
import hashlib
import json
import re

import numpy as np

from trajectory_prediction.robustness.source_gaussian import SEEDS, validate_packets


def _number(key,seed,channel):
    raw=json.dumps({'schema':'realclock_outlier_packet_b3b2b_v1','source_key_digest':key,
                    'noise_seed':seed,'channel':channel},sort_keys=True,separators=(',',':'),
                   ensure_ascii=False,allow_nan=False).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')


def packet_outlier(key,seed):
    if (not isinstance(key,str) or re.fullmatch('[0-9a-f]{64}',key) is None
            or type(seed) is not int or seed not in SEEDS):
        raise ValueError('exact source digest and frozen noise seed required')
    flag=_number(key,seed,'occurrence') < 2**64//20
    theta=2*np.pi*(_number(key,seed,'direction')/2**64)
    return flag,5*np.array([np.cos(theta),np.sin(theta)],np.float64) if flag else np.zeros(2,np.float64)


def outlier_table(packets,seed):
    values=[packet_outlier(p['key_digest'],seed) for p in validate_packets(packets)]
    return {'injected':np.array([v[0] for v in values],dtype=bool),
            'outliers':np.stack([v[1] for v in values])}
