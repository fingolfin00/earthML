from typing import Any
from collections.abc import Sequence
from pathlib import Path

from .settings import Settings


def get_experiment_configs(
    experiments_root: str | Path,
    **filters: Any,
) -> list[Settings]:
    unknown = set(filters) - Settings.field_names()
    if unknown:
        raise ValueError(f"Unknown Settings fields: {unknown}")

    config_paths = list(Path(experiments_root).rglob("config.json"))

    matching_settings: list[Settings] = []

    for config_path in config_paths:
        s = Settings.from_json(config_path)

        for name, expected in filters.items():
            if expected is None:
                continue

            actual = getattr(s, name)

            if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
                if actual not in expected:
                    break
            else:
                if actual != expected:
                    break
        else:
            matching_settings.append(s)

    return matching_settings
