import os, logging, warnings
from pathlib import Path
from typing import Literal
from dataclasses import dataclass

from ...misc import Dask
from ...logging import configure_logging, get_logger, LoggingConfig

def configure_warnings_and_logging(log_config: LoggingConfig) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
    logging.getLogger("distributed.worker").setLevel(logging.ERROR)
    logging.getLogger("distributed.nanny").setLevel(logging.ERROR)
    configure_logging(log_config)

def configure_ca_bundle() -> None:
    bundle = Path.home() / "certs" / "earthml-ca-bundle.pem"
    if bundle.is_file():
        os.environ["REQUESTS_CA_BUNDLE"] = str(bundle)
        os.environ["SSL_CERT_FILE"] = str(bundle)

@dataclass
class Runtime:
    """Utility class for setting up MLBC manager's runtime"""
    dask_workers: int | None = None
    memory_limit: str = "auto"
    nanny: bool = True
    needs_ca_bundle: bool = False
    log_level: Literal["info", "debug"] = "info"

    def start(self) -> Dask:
        log_config = LoggingConfig(level=self.log_level)
        configure_warnings_and_logging(log_config)
        if self.needs_ca_bundle:
            configure_ca_bundle()
        d = Dask(
            n_workers=self.dask_workers,
            memory_limit=self.memory_limit,
            nanny=self.nanny,
        )
        d.start()
        get_logger().info("Dask dashboard: %s", d.client.dashboard_link)
        return d
