"""Source-clock CV, alpha-beta and Joseph-form Kalman estimators; no label inputs."""
import time

import numpy as np

STRATEGIES = ("raw_cv", "alpha_beta", "kf_cv", "kf_nis")
SETTINGS = {"alpha": .6, "beta": .2, "measurement_variance": .25,
            "acceleration_spectral_density": 1., "nis_threshold": 9.21, "maximum_r_scale": 25.}


def _transition(dt):
    eye = np.eye(2)
    f = np.block([[eye, dt * eye], [np.zeros((2, 2)), eye]])
    q = SETTINGS["acceleration_spectral_density"] * np.block([
        [dt ** 3 / 3 * eye, dt ** 2 / 2 * eye], [dt ** 2 / 2 * eye, dt * eye]])
    return f, q


class Estimator:
    def __init__(self, strategy, first, second):
        if strategy not in STRATEGIES or first.source_step >= second.source_step:
            raise ValueError("valid strategy and increasing source steps required")
        dt = (second.source_step - first.source_step) * .5
        self.strategy, self.source_step = strategy, second.source_step
        self.x = np.concatenate((second.position, (second.position - first.position) / dt))
        r = np.eye(2) * SETTINGS["measurement_variance"]
        self.p = np.block([[r, r / dt], [r / dt, 2 * r / dt ** 2]])
        self.last_raw = second.position.copy()
        self.nis, self.r_scale = 0., 1.

    def receive(self, packet):
        if packet.source_step <= self.source_step:
            raise ValueError("source time must increase; out-of-order packets require a separate protocol")
        dt = (packet.source_step - self.source_step) * .5
        if self.strategy == "raw_cv":
            self.x = np.concatenate((packet.position, (packet.position - self.last_raw) / dt))
        else:
            f, q = _transition(dt)
            predicted = f @ self.x
            residual = packet.position - predicted[:2]
            if self.strategy == "alpha_beta":
                self.x = predicted
                self.x[:2] += SETTINGS["alpha"] * residual
                self.x[2:] += SETTINGS["beta"] * residual / dt
            else:
                p = f @ self.p @ f.T + q
                r = np.eye(2) * SETTINGS["measurement_variance"]
                s = p[:2, :2] + r
                self.nis = float(residual @ np.linalg.solve(s, residual))
                self.r_scale = (min(SETTINGS["maximum_r_scale"], max(1., self.nis / SETTINGS["nis_threshold"]))
                                if self.strategy == "kf_nis" else 1.)
                r *= self.r_scale
                gain = np.linalg.solve(p[:2, :2] + r, p[:2, :]).T
                self.x = predicted + gain @ residual
                identity = np.eye(4)
                identity[:, :2] -= gain
                self.p = identity @ p @ identity.T + gain @ r @ gain.T
                self.p = (self.p + self.p.T) / 2
        self.source_step, self.last_raw = packet.source_step, packet.position.copy()

    def predict(self, now):
        if now < self.source_step:
            raise ValueError("query cannot precede latest source")
        dt = (now - self.source_step) * .5
        f, q = _transition(dt)
        state = f @ self.x
        covariance = (f @ self.p @ f.T + q)[:2, :2]
        lead = np.arange(1, 13) * .5 - now * .5
        forecast = state[:2] + lead[:, None] * state[2:]
        return forecast, state[:2].copy(), covariance


def run_packets(packets, strategy):
    packets = tuple(packets)
    arrival_keys = [(p.available_step, p.source_step) for p in packets]
    if not packets or arrival_keys != sorted(arrival_keys):
        raise ValueError("packets must be arrival ordered")
    initial = [p for p in packets if p.available_step <= 0]
    if len(initial) < 2:
        raise ValueError("at least two arrived historical packets required")
    model = Estimator(strategy, initial[0], initial[1])
    for packet in initial[2:]:
        model.receive(packet)
    pending = [p for p in packets if p.available_step > 0]
    index, updates = 0, 0
    forecasts, positions, covariances, fresh, scales, times, ages = [], [], [], [], [], [], []
    for now in range(1, 12):
        begun = time.perf_counter_ns()
        count = 0
        while index < len(pending) and pending[index].available_step <= now:
            packet = pending[index]
            if packet.source_step > now:
                raise ValueError("future source leaked into observation")
            model.receive(packet)
            index, count = index + 1, count + 1
        forecast, position, covariance = model.predict(now)
        times.append((time.perf_counter_ns() - begun) / 1e6)
        forecasts.append(forecast)
        positions.append(position)
        covariances.append(covariance)
        fresh.append(count > 0)
        scales.append(model.r_scale)
        ages.append((now - model.source_step) * .5)
        updates += count
    return {"forecasts": np.asarray(forecasts), "positions": np.asarray(positions),
            "position_covariances": np.asarray(covariances), "fresh": np.asarray(fresh),
            "r_scales": np.asarray(scales), "ages": np.asarray(ages),
            "latency_ms": np.asarray(times), "measurement_updates": updates}
