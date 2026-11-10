"""A fixed causal turning/low-speed prior for allowing the straight CA expert."""
import numpy as np

from trajectory_prediction.robustness.filters import Estimator

GUARD_SETTINGS = {'window': 4, 'minimum_speed': .5, 'maximum_yaw_change_deg': 15.}


def build_guard(packets):
    packets = tuple(packets)
    if ([p.source_step for p in packets] != list(range(-4, 12))
            or any(p.source_step != p.available_step for p in packets)):
        raise ValueError('guard requires regular complete zero-delay packets')
    model = Estimator('kf_nis', packets[0], packets[1])
    velocities = [model.x[2:].copy()]
    for packet in packets[2:5]:
        model.receive(packet)
        velocities.append(model.x[2:].copy())
    allowed, yaws, speeds = [], [], []
    for packet in packets[5:]:
        model.receive(packet)
        velocities.append(model.x[2:].copy())
        window = np.array(velocities[-GUARD_SETTINGS['window']:])
        minimum = float(np.linalg.norm(window, axis=1).min())
        angle = np.arctan2(window[:, 1], window[:, 0])
        delta = (np.diff(angle)+np.pi)%(2*np.pi)-np.pi
        yaw = float(np.degrees(np.abs(delta).sum()))
        allowed.append(minimum >= GUARD_SETTINGS['minimum_speed'] and yaw < GUARD_SETTINGS['maximum_yaw_change_deg'])
        yaws.append(yaw)
        speeds.append(minimum)
    return {'ca_allowed': np.array(allowed, bool), 'guard_yaw': np.array(yaws, np.float32),
            'guard_min_speed': np.array(speeds, np.float32)}
