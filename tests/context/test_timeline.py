import pytest

from trajectory_prediction.context.schema import FrameRef
from trajectory_prediction.context.timeline import latest_available


def frame(token, source, available, channel="CAM_FRONT"):
    return FrameRef(token, channel, source, available)


def test_availability_excludes_delayed_and_future_frames():
    frames = (
        frame("old", 100, 100),
        frame("delayed", 200, 400),
        frame("future", 301, 301),
    )
    selected = latest_available(frames, "CAM_FRONT", 300)
    assert selected.token == "old"


def test_inclusive_boundary_channel_and_staleness():
    frames = (frame("camera", 100, 200), frame("radar", 199, 199, "RADAR_FRONT"))
    assert latest_available(frames, "CAM_FRONT", 200, max_age_us=100).token == "camera"
    assert latest_available(frames, "CAM_FRONT", 201, max_age_us=100) is None
    assert latest_available(frames, "LIDAR_TOP", 200) is None


def test_order_does_not_change_latest_source_selection():
    frames = (frame("a", 100, 290), frame("b", 200, 250))
    assert latest_available(frames, "CAM_FRONT", 300).token == "b"
    assert latest_available(tuple(reversed(frames)), "CAM_FRONT", 300).token == "b"


@pytest.mark.parametrize("source,available", [(-1, 0), (100, 99), (True, 100), (1.5, 10)])
def test_frame_rejects_invalid_timestamps(source, available):
    with pytest.raises(ValueError):
        frame("bad", source, available)



def test_tied_source_timestamp_uses_token_independent_of_input_order():
    frames = (frame("a", 100, 200), frame("z", 100, 150))
    for ordered in (frames, tuple(reversed(frames))):
        assert latest_available(iter(ordered), "CAM_FRONT", 200).token == "z"


def test_empty_frames_and_zero_age_boundary():
    assert latest_available((), "CAM_FRONT", 0) is None
    assert latest_available((frame("now", 0, 0),), "CAM_FRONT", 0, max_age_us=0).token == "now"
    assert latest_available((frame("old", 0, 0),), "CAM_FRONT", 1, max_age_us=0) is None


@pytest.mark.parametrize("token,channel", [("", "CAM_FRONT"), (None, "CAM_FRONT"), ("a", ""), ("a", 1)])
def test_frame_rejects_invalid_identity(token, channel):
    with pytest.raises(ValueError):
        FrameRef(token, channel, 0, 0)


@pytest.mark.parametrize("value", [-1, True, 1.5, None])
@pytest.mark.parametrize("argument", ["timestamp_us", "max_age_us"])
def test_query_rejects_invalid_times(value, argument):
    kwargs = {"timestamp_us": 100, "max_age_us": 100, argument: value}
    with pytest.raises(ValueError):
        latest_available((), "CAM_FRONT", **kwargs)


@pytest.mark.parametrize("channel", ["", None, 1])
def test_query_rejects_invalid_channel(channel):
    with pytest.raises(ValueError):
        latest_available((), channel, 0)


@pytest.mark.parametrize("available", [-1, True, 1.5])
def test_frame_rejects_invalid_availability(available):
    with pytest.raises(ValueError):
        frame("bad", 0, available)
