"""One actual state update and feature construction, independent of cached extras."""

import numpy as np
import torch


def new_state(data, c, i):
    history = data.arrays["history"][i]
    heading = np.zeros(2, np.float32)
    for velocity in np.diff(history, axis=0)/np.float32(.5):
        speed = np.linalg.norm(velocity)
        if speed >= .5:
            heading = velocity/speed
    return {"key": (c, i), "next_step": 0, "heading": heading,
        "previous_velocity": (history[-1]-history[-2])/np.float32(.5),
        "cv_error": 0., "map_error": 0., "cv_has": False, "map_has": False,
        "last_cv": None, "last_map": None, "last_valid": False}


def stream_inputs(data, c, i, k, state, device):
    if state["key"] != (c, i) or state["next_step"] != k or not 0 <= k < 11:
        raise ValueError("online horizon state/key/step mismatch")
    a = data.arrays
    position, velocity = a["position"][i, k], a["velocity"][i, k]
    valid = bool(a["valid"][c, i, k])
    if k:
        error = float(np.linalg.norm(state["last_cv"][k]-position))
        state["cv_error"] = .8*state["cv_error"]+.2*error if state["cv_has"] else error
        state["cv_has"] = True
        if state["last_valid"] and valid:
            error = float(np.linalg.norm(state["last_map"][k]-position))
            state["map_error"] = .8*state["map_error"]+.2*error if state["map_has"] else error
            state["map_has"] = True
    if not valid:
        state["map_has"], state["map_error"] = False, 0.
    speed = np.linalg.norm(velocity)
    if speed >= .5:
        state["heading"] = velocity/speed
    heading = state["heading"]
    lateral = np.array([-heading[1], heading[0]], np.float32)
    acceleration = (velocity-state["previous_velocity"])/np.float32(.5)
    cv, candidate = a["cv"][i, k], a["map"][c, i, k]
    delta = candidate-cv
    mask = np.arange(12) >= k+1
    mean = (delta*mask[:, None]).sum(0, dtype=np.float64)/mask.sum()
    direction = np.clip([np.dot(acceleration, heading)/10., np.dot(acceleration, lateral)/10.,
        np.dot(mean, heading)/20., np.dot(mean, lateral)/20.], -5, 5)
    feedback = [min(np.log1p(state["cv_error"]), 5.), min(np.log1p(state["map_error"]), 5.), state["cv_has"], state["map_has"]]
    features = np.empty((1, 12, 19), np.float32)
    features[0, :, :8] = a["quality"][c, i, k]
    features[0, :, 8:12], features[0, :, 12:16] = direction, feedback
    features[0, :, 16] = np.maximum((np.arange(12)-k)*.5, 0)/6.
    features[0, :, 17] = np.clip((delta*heading).sum(-1)/20., -5, 5)
    features[0, :, 18] = np.clip((delta*lateral).sum(-1)/20., -5, 5)
    state.update(next_step=k+1, previous_velocity=velocity.copy(), last_cv=cv.copy(), last_map=candidate.copy(), last_valid=valid)
    values = {"features": features, "quality": np.ascontiguousarray(a["quality"][c, i, k][None]),
        "cv": np.ascontiguousarray(cv[None]), "map": np.ascontiguousarray(candidate[None]),
        "valid": np.array([valid]), "mask": mask[None]}
    return {name: torch.from_numpy(value).to(device) for name, value in values.items()}
