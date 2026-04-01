import os, psutil, multiprocessing, tempfile, logging
import numpy as np

import dask
from dask.distributed import Client, LocalCluster


class Dask:
    """
    Class to initialize a Dask cluster
    """
    def __init__(self, base_port=8787, n_workers=None, processes=True, nanny=True, memory_limit="auto"):
        self.base_port = base_port
        self.n_workers = n_workers
        self.processes = processes
        self.nanny = nanny
        self.memory_limit = memory_limit

        self.cluster = None
        self.client = None

    def start(self):
        logging.getLogger("tornado.application").setLevel(logging.ERROR)
        logging.getLogger("bokeh").setLevel(logging.ERROR)

        candidates = [
            os.environ.get("TMPDIR"),
            tempfile.gettempdir(),
            os.path.expanduser("~/.dask-tmp"),
            f"/scratch/{os.environ.get('USER')}",
            "/scratch",
        ]
        local_dir = next((p for p in candidates if p and os.path.exists(p)), tempfile.gettempdir())
        os.makedirs(local_dir, exist_ok=True)

        # Force a spawn start method to avoid issues with forking and HDF5 in some environments
        multiprocessing.set_start_method("spawn", force=True)

        n_cores = multiprocessing.cpu_count()
        total_mem_gb = psutil.virtual_memory().total / 1e9
        n_workers = self.n_workers
        if n_workers is None:
            n_workers = min(n_cores, max(1, int(total_mem_gb // 4)))

        self.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=self.processes,
            timeout="600s",
            heartbeat_interval="10s",
            memory_limit=self.memory_limit,
            local_directory=local_dir,
            dashboard_address=f":{self.base_port}",
            nanny=self.nanny,
        )
        self.client = Client(self.cluster)

        info = self.client.scheduler_info()
        n_active_workers = len(info["workers"])
        workers_mem_limit_gb = []
        for w in info["workers"].values():
            workers_mem_limit_gb.append(w["memory_limit"] / 1024**3)
        workers_mem_limit_gb = np.array(workers_mem_limit_gb)

        # Register cf_xarray
        self.client.run(lambda: __import__("cf_xarray"))

        import socket
        print(f"Dask dashboard running on {socket.gethostname()}:{self.cluster.scheduler.services['dashboard'].port}")
        print(f"Cores: {n_cores}, Mem: {total_mem_gb} GB -> Dask active workers: {n_active_workers} (requested: {n_workers})")
        print(f"Dask memory per worker (avg): {workers_mem_limit_gb.mean():.2f} GB")
        print(f"Write Dask local files in {local_dir}")

        return self

    def close(self):
        """
        Gracefully close Dask client and cluster, with retries to handle transient issues.
        """
        if self.client is not None:
            try:
                self.client.close(timeout=10)
            except Exception:
                pass
            self.client = None

        if self.cluster is not None:
            try:
                self.cluster.close(timeout=10)
            except Exception:
                pass
            self.cluster = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # don't suppress exceptions
