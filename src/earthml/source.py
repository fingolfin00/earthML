import time
from abc import ABC, abstractmethod
import xarray as xr
import cf_xarray as cfxr
from pathlib import Path
from functools import partial
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor
from functools import partial
# Local imports
from .dataclasses import DataSelection

class SourceRegistry:
    def __init__ (self, source_name: str):
        self.class_registry = {
            "juno-grib": JunoGribSource
        }
        self.source_name = source_name

    def get_class (self):
        return self.class_registry.get(self.source_name)

class LocalSource (ABC):
    def __init__ (
        self,
        data_selection: DataSelection,
        root_path: str,
    ):
        self.ds = None
        self.path = Path(root_path)
        self.data_selection = data_selection
        self.date_range = xr.date_range(
            start=self.data_selection.period.start,
            end=self.data_selection.period.end,
            freq=self.data_selection.period.freq
        )
        self.file_paths = {}

    @staticmethod
    def _preprocess(ds: xr.Dataset, data: DataSelection) -> xr.DataArray:
        # print(f"ds.dims: {ds.dims}")
        # print(f"ds.coords: {ds.coords}")
        time_dim = ds.cf['time'].name
        # print(f"Time dimension: {time_dim}")
        if time_dim not in ds.dims:
            # print("Expanding time dimension")
            ds = ds.cf.expand_dims(time=[ds.cf["time"].values])
            ds = ds.assign_coords(
                **{time_dim: ds[time_dim].assign_attrs(standard_name="time", axis="T")}
            )
        # if time_dim != "time" and "time" in ds.dims:
        #     ds = ds.drop_vars("time")
        # if time_dim == "valid_time":
        #     ds = ds.rename(valid_time="time")
        return ds[data.variable.name].cf.sel(
            longitude=slice(*data.region.lon), latitude=slice(*data.region.lat),
            # time=slice(data.period.start, data.period.end)
        )
    
    def load (self) -> xr.DataArray:
        """Get data only if it hasn't been done yet"""
        if not self.ds:
            print("Get data...")
            t0 = time.time()
            self.ds = self._get_data()
            print(f"open_mfdataset time: {time.time() - t0:.2f}s")
            # t0 = time.time()
            # self.ds = self._get_data_processpool(max_workers=36)
            # print(f"processpool time: {time.time() - t0:.2f}s")
        return self.ds

    def reload (self) -> xr.DataArray:
        """Force data reload"""
        self.ds = self._get_data()
        return self.ds

    @abstractmethod
    def _get_data_filenames (self) -> dict:
        """
        Get the local data filenames for the given data selection.
        """
        pass
    
    @abstractmethod
    def _get_data (self) -> xr.DataArray:
        """
        Get the local data for the given data selection.
        """
        pass

class JunoGribSource (LocalSource):
    """
    Collect Juno grib data for the given data selection.
    """
    def __init__ (
        self,
        data_selection: DataSelection,
        root_path: str,
        file_path_date_format: str,
        file_header: str,
        file_date_format: str,
        lead_time: timedelta,
        minus_timedelta: timedelta = None,
        plus_timedelta: timedelta = None
    ):
        super().__init__ (data_selection, root_path)
        self.file_paths = self._get_data_filenames(
            file_path_date_format,
            file_header,
            file_date_format,
            lead_time,
            minus_timedelta,
            plus_timedelta
        )

    def _get_data_filenames(
        self,
        file_path_date_format: str,
        file_header: str,
        file_date_format: str,
        lead_time: timedelta,
        minus_timedelta: timedelta = None,
        plus_timedelta: timedelta = None
    ) -> dict:
        """Get the data filenames for the given data selection."""

        samples, missed_samples, plus_samples, minus_samples = {}, [], [], []

        print(f"Date range length: {len(self.date_range)}")
        for date in self.date_range:
            forward_date = date + lead_time
            data_path = self.path.joinpath(date.strftime(file_path_date_format))
            data_glob = f"{file_header}{date.strftime(file_date_format)}{forward_date.strftime(file_date_format)}*"

            # files_exact = [p for p in data_path.glob(data_glob) if p.is_file()]
            files_exact = sorted(
                (p for p in data_path.glob(data_glob) if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True  # newest first
            )

            if len(files_exact) > 1:
                print(f"{date}: {len(files_exact)} matches, keeping {files_exact[:1]}")
                files_exact = files_exact[:1]  # keep latest only, still a list
            # print(files_exact)

            # Try direct match first
            if files_exact:
                samples[date] = files_exact[0]
                continue

            # Fallback search
            found = False
            if minus_timedelta and plus_timedelta:
                # print(minus_timedelta, plus_timedelta)
                for delta, store in [
                    (-minus_timedelta, minus_samples),
                    (plus_timedelta, plus_samples),
                ]:
                    test_date = forward_date + delta
                    test_glob = f"{file_header}{date.strftime(file_date_format)}{test_date.strftime(file_date_format)}*"
                    test_files = [p for p in data_path.glob(test_glob) if p.is_file()]
                    # print(f"New file {delta}: {test_files}")
                    if test_files:
                        samples[date] = test_files[0]
                        # samples.extend(test_files)
                        store.append(date)
                        found = True
                        break

            if not found:
                print(f"Missed sample: {date}")
                missed_samples.append(date)

        return {
            "samples": samples,
            "missed_samples": missed_samples,
            "plus_samples": plus_samples,
            "minus_samples": minus_samples,
        }

    # Worker function to open a single file
    def _open_one(self, fp, backend_kwargs):
        try:
            ds = xr.open_dataset(
                fp,
                engine="cfgrib",
                indexpath="",
                backend_kwargs=backend_kwargs,
                decode_timedelta=True,
                decode_cf=True,
            )
            return self._preprocess(ds, data=self.data_selection)
        except Exception as e:
            import traceback
            print(f"Failed to open {fp}: {type(e).__name__} – {e}")
            traceback.print_exc(limit=1)
            try:
                ds = xr.open_dataset(
                    fp,
                    engine="cfgrib",
                    indexpath="",
                    # backend_kwargs=backend_kwargs,
                    decode_timedelta=True,
                    decode_cf=True,
                )
                print(f"{fp}: opened without filter_by_keys due to {e}")
                return self._preprocess(ds, data=self.data_selection)
            except Exception:
                print(f"Skipping {fp}")
                return None


    def _get_data (self) -> xr.DataArray:
        # years = [str(date.year) for date in xr.date_range(start=data.period.start, end=data.period.end, freq='YS')]
        samples = list(self.file_paths['samples'].values())
        print(f"Samples: {len(samples)}, minus: {len(self.file_paths['minus_samples'])}, plus: {len(self.file_paths['plus_samples'])}, missed: {len(self.file_paths['missed_samples'])}")
        return xr.open_mfdataset(
            samples,
            combine="by_coords",
            coords=["time"],
            compat="override" if (self.file_paths['minus_samples'] or self.file_paths['plus_samples']) else "no_conflicts",
            engine="cfgrib",
            indexpath="",
            chunks={},  # should make if faster
            parallel=True,
            decode_timedelta=True,
            backend_kwargs={
                "filter_by_keys": {"cfVarName": self.data_selection.variable.name},
            },
            preprocess=partial(self._preprocess, data=self.data_selection),
            decode_cf=True,
            errors="warn",
        )

    def _get_data_processpool (self, max_workers: int = 8) -> xr.DataArray:
        """
        Drop-in alternative to _get_data() that opens GRIB files in parallel
        using multiple processes (since cfgrib is not thread-safe).
        """
        samples = self.file_paths['samples']
        print(f"Samples: {len(samples)}, minus: {len(self.file_paths['minus_samples'])}, "
            f"plus: {len(self.file_paths['plus_samples'])}, missed: {len(self.file_paths['missed_samples'])}")

        compat = "override" if (
            len(self.file_paths['minus_samples']) or len(self.file_paths['plus_samples'])
        ) else "no_conflicts"

        backend_kwargs = {
            # "filter_by_keys": {"cfVarName": self.data_selection.variable.name},
        }

        # Load files in parallel using separate processes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            datasets = list(executor.map(self._open_one, samples, backend_kwargs))

        # Filter out any None results
        datasets = [ds for ds in datasets if ds is not None]

        if not datasets:
            raise RuntimeError("No datasets could be opened — all failed.")

        # Combine along coordinates (like open_mfdataset does)
        combined = xr.combine_by_coords(datasets, compat=compat, combine_attrs="override")

        return combined

