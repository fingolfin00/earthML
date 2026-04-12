import getpass
import os
import tempfile
from pathlib import Path


_EARTHKIT_CONFIG_FILE_ENV = "EARTHKIT_DATA_CONFIG_FILE"
_EARTHML_MANAGED_CONFIG_ENV = "EARTHML_MANAGED_EARTHKIT_CONFIG"


def configure_sources_runtime() -> None:
    _configure_earthkit_config_file()


def _configure_earthkit_config_file() -> None:
    current = os.environ.get(_EARTHKIT_CONFIG_FILE_ENV)
    managed = os.environ.get(_EARTHML_MANAGED_CONFIG_ENV) == "1"

    if current and not managed:
        return

    config_root = (
        Path(tempfile.gettempdir())
        / f"earthml-{getpass.getuser()}"
        / "earthkit-config"
        / f"pid-{os.getpid()}"
    )
    config_root.mkdir(parents=True, exist_ok=True)

    os.environ[_EARTHKIT_CONFIG_FILE_ENV] = str(config_root / "config.yaml")
    os.environ[_EARTHML_MANAGED_CONFIG_ENV] = "1"
