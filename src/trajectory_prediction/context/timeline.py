from collections.abc import Iterable

from trajectory_prediction.context.schema import FrameRef, validate_timestamp_us


def latest_available(
    frames: Iterable[FrameRef],
    channel: str,
    timestamp_us: int,
    *,
    max_age_us: int = 2_000_000,
) -> FrameRef | None:
    validate_timestamp_us(timestamp_us)
    validate_timestamp_us(max_age_us)
    if not isinstance(channel, str) or not channel:
        raise ValueError("channel must be a nonempty string")
    candidates = (
        item for item in frames
        if item.channel == channel
        and item.available_timestamp_us <= timestamp_us
        and timestamp_us - item.source_timestamp_us <= max_age_us
    )
    return max(candidates, key=lambda item: (item.source_timestamp_us, item.token), default=None)
