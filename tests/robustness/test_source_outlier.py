import hashlib
import json

import numpy as np
import pytest


def reference(key,seed,channel):
    raw=json.dumps({'schema':'realclock_outlier_packet_b3b2b_v1','source_key_digest':key,
                    'noise_seed':seed,'channel':channel},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')


def test_exact_hash_reference_and_rng_preserved():
    from trajectory_prediction.robustness.source_outlier import packet_outlier
    before=np.random.get_state()
    seen=set()
    for i in range(120):
        key=f'{i:064x}'
        for seed in (2027,2028,2029):
            flag,xy=packet_outlier(key,seed)
            expected=reference(key,seed,'occurrence') < 2**64//20
            assert flag is expected
            theta=2*np.pi*(reference(key,seed,'direction')/2**64)
            np.testing.assert_array_equal(xy,5*np.array([np.cos(theta),np.sin(theta)]) if flag else np.zeros(2))
            if flag:seen.add(seed);assert np.linalg.norm(xy)==pytest.approx(5.)
            np.testing.assert_array_equal(xy,packet_outlier(key,seed)[1])
    assert seen=={2027,2028,2029}
    after=np.random.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])


@pytest.mark.parametrize('key,seed',[('A'*64,2027),('0'*63,2027),('0'*64,True),('0'*64,1)])
def test_strict_key_seed(key,seed):
    from trajectory_prediction.robustness.source_outlier import packet_outlier
    with pytest.raises(ValueError):packet_outlier(key,seed)


def test_occurrence_cutoff_is_strict(monkeypatch):
    from trajectory_prediction.robustness import source_outlier as mod
    monkeypatch.setattr(mod,'_number',lambda key,seed,channel: 2**64//20)
    assert mod.packet_outlier('0'*64,2027)[0] is False
    monkeypatch.setattr(mod,'_number',lambda key,seed,channel: 2**64//20-1)
    assert mod.packet_outlier('0'*64,2027)[0] is True
