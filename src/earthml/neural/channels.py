from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelInfo:
    name: str
    variable: str | None = None
    temporal_offset: int | None = None
    normalize: bool = True


def validate_channels(
    channels: tuple[ChannelInfo, ...],
) -> None:
    names = [channel.name for channel in channels]

    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate channel names: {names}")

    for channel in channels:
        if not channel.name:
            raise ValueError("Channel name cannot be empty")

        if channel.temporal_offset is not None:
            if type(channel.temporal_offset) is not int:
                raise ValueError(f"Invalid temporal offset for {channel.name!r}")

            if channel.variable is None:
                raise ValueError(f"Temporal channel {channel.name!r} must specify a variable")
