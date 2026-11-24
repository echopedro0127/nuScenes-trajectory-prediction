"""Versioned, source-packet-keyed standard normals; no global RNG state."""
import hashlib
import json
import re

import numpy as np

from trajectory_prediction.analysis.track_timeline import digest

SEEDS = (2027,2028,2029)
BASE = ('schema','dataset_release','role','scene','log','instance','annotation_token')


def normal_vector(source_digest,seed):
    if (not isinstance(source_digest,str) or re.fullmatch('[0-9a-f]{64}',source_digest) is None
            or type(seed) is not int or seed not in SEEDS):
        raise ValueError('exact source digest and frozen noise seed required')
    key = {'schema':'realclock_gaussian_packet_b3b2a_v1','source_key_digest':source_digest,
           'noise_seed':seed,'channel':'world_xy_standard_normal'}
    raw = json.dumps(key,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    number = int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')
    return np.random.Generator(np.random.PCG64(number)).standard_normal(size=2,dtype=np.float64)


def validate_packets(packets):
    if not isinstance(packets,list) or not packets:raise ValueError('nonempty source manifest required')
    seen,identities = set(),set()
    for p in packets:
        if (not isinstance(p,dict) or set(p) != set(BASE)|{'key_digest','source_us'}
                or any(not isinstance(p[k],str) or not p[k].strip() for k in BASE)
                or p['schema'] != 'realclock_source_packet_v1'
                or p['role'] not in ('independent_fit','independent_select')
                or type(p['source_us']) is not int or not 0 < p['source_us'] < 2**63
                or p['key_digest'] != digest({k:p[k] for k in BASE})
                or p['key_digest'] in seen or (p['role'],p['annotation_token']) in identities):
            raise ValueError('invalid or duplicate source packet manifest')
        seen.add(p['key_digest']);identities.add((p['role'],p['annotation_token']))
    return sorted(packets,key=lambda p:p['key_digest'])


def normal_table(packets,seed):
    return np.stack([normal_vector(p['key_digest'],seed) for p in validate_packets(packets)])


def window_indices(packets,windows):
    ordered = validate_packets(packets)
    lookup = {(p['role'],p['annotation_token']):i for i,p in enumerate(ordered)}
    rows = []
    for w in windows:
        if (any(k not in w for k in ('role','instance','scene','log','tokens','time_us'))
                or len(w['tokens']) != len(w['time_us']) or not w['tokens']):
            raise ValueError('invalid window source fields')
        row = []
        for token,time in zip(w['tokens'],w['time_us'],strict=True):
            i = lookup.get((w['role'],token))
            if i is None:raise ValueError('foreign window packet')
            p = ordered[i]
            if p['source_us'] != time or any(p[k] != w[k] for k in ('role','instance','scene','log')):
                raise ValueError('window packet clock/identity mismatch')
            row.append(i)
        rows.append(row)
    if not rows or len({len(r) for r in rows}) != 1:raise ValueError('uniform nonempty windows required')
    return np.array(rows,np.int32)
