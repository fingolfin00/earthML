import os, logging, warnings
from pathlib import Path
from dataclasses import dataclass
from rich import print

from earthml.utils import Dask

def configure_warnings_and_logging() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)

def configure_ca_bundle() -> None:
    bundle = Path.home() / "certs" / "earthml-ca-bundle.pem"
    if bundle.is_file():
        os.environ["REQUESTS_CA_BUNDLE"] = str(bundle)
        os.environ["SSL_CERT_FILE"] = str(bundle)

@dataclass
class Runtime:
    """Utility class for setting up launcher's runtime"""
    dask_workers: int | None = None
    needs_ca_bundle: bool = False

    def start(self) -> Dask:
        configure_warnings_and_logging()
        if self.needs_ca_bundle:
            configure_ca_bundle()
        d = Dask(n_workers=self.dask_workers)
        d.start()
        print("Dask dashboard:", d.client.dashboard_link)
        return d
