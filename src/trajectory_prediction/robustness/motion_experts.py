"""Causal scalar-speed motion priors, integrated with a nonnegative speed floor."""
import numpy as np

from trajectory_prediction.robustness.filters import Estimator

EXPERTS = ('bounded_ca', 'brake_stop', 'hold_position')
MOTION_SETTINGS = {'acceleration_min': -4., 'acceleration_max': 2., 'braking': -2.,
                   'heading_speed_min': .1, 'velocity_window': 4, 'dt': .5}


def bounded_path(position, velocity, acceleration, times):
    position, velocity, times = np.asarray(position), np.asarray(velocity), np.asarray(times)
    if (position.shape != (2,) or velocity.shape != (2,) or times.ndim != 1 or (times < 0).any()
            or not all(np.isfinite(v).all() for v in (position, velocity, times)) or not np.isfinite(acceleration)):
        raise ValueError('finite XY state and nonnegative forecast times required')
    speed = float(np.linalg.norm(velocity))
    if speed < MOTION_SETTINGS['heading_speed_min']:
        return np.repeat(position[None], len(times), axis=0)
    a = float(np.clip(acceleration, MOTION_SETTINGS['acceleration_min'], MOTION_SETTINGS['acceleration_max']))
    elapsed = np.minimum(times, speed/-a) if a < 0 else times
    distance = speed*elapsed + .5*a*elapsed**2
    return position + distance[:, None]*(velocity/speed)


def build_motion_experts(packets):
    packets = tuple(packets)
    if ([p.source_step for p in packets] != list(range(-4, 12))
            or any(p.source_step != p.available_step for p in packets)):
        raise ValueError('motion priors require regular complete zero-delay packets')
    model = Estimator('kf_nis', packets[0], packets[1])
    speeds = [np.linalg.norm(model.x[2:])]
    for packet in packets[2:5]:
        model.receive(packet)
        speeds.append(np.linalg.norm(model.x[2:]))
    forecasts = []
    for now, packet in enumerate(packets[5:], 1):
        model.receive(packet)
        speeds.append(np.linalg.norm(model.x[2:]))
        window = np.array(speeds[-MOTION_SETTINGS['velocity_window']:])
        t = np.arange(len(window))*.5
        centered = t-t.mean()
        acceleration = float(centered@window/(centered@centered))
        lead = np.maximum(np.arange(1, 13)*.5-now*.5, 0)
        position, velocity = model.x[:2], model.x[2:]
        forecasts.append([bounded_path(position, velocity, acceleration, lead),
                          bounded_path(position, velocity, MOTION_SETTINGS['braking'], lead),
                          np.repeat(position[None], 12, axis=0)])
    result = np.array(forecasts, np.float32)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite bounded motion forecasts')
    return result
