import os, logging, warnings
from pathlib import Path
from dataclasses import dataclass

from ...misc import Dask
from ...logging import configure_logging, get_logger

def configure_warnings_and_logging() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
    logging.getLogger("distributed.worker").setLevel(logging.ERROR)
    logging.getLogger("distributed.nanny").setLevel(logging.ERROR)
    configure_logging()

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

    def start(self) -> Dask:
        configure_warnings_and_logging()
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
