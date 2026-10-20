from dataclasses import dataclass


def validate_timestamp_us(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("timestamp must be a nonnegative integer in microseconds")


@dataclass(frozen=True)
class FrameRef:
    token: str
    channel: str
    source_timestamp_us: int
    available_timestamp_us: int

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("frame token must be a nonempty string")
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("frame channel must be a nonempty string")
        validate_timestamp_us(self.source_timestamp_us)
        validate_timestamp_us(self.available_timestamp_us)
        if self.available_timestamp_us < self.source_timestamp_us:
            raise ValueError("availability cannot precede acquisition")
