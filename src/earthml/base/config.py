from typing import Sequence
from pathlib import Path

from .settings import Settings


def get_experiment_configs(
    experiments_root: str | Path,
    variables: Sequence | None = None,
    regions: Sequence | None = None,
) -> list[Settings]:
    config_paths = list(Path(experiments_root).rglob("config.json"))

    matching_settings: list[Settings] = []

    for config_path in config_paths:
        s = Settings.from_json(config_path)

        if variables is not None and s.var_fc not in variables:
            continue

        if regions is not None and s.region_name not in regions:
            continue

        matching_settings.append(s)

    return matching_settings
