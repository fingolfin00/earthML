import time
from abc import ABC, abstractmethod
# import dask
import cf_xarray
import xarray as xr
from pathlib import Path
from functools import partial
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from zarr.codecs import BloscCodec
# Local imports
from .dataclasses import DataSelection

class SourceRegistry:
    def __init__ (self, source_name: str):
        self.class_registry = {
            "juno-local": JunoLocalSource,
            "earthkit": EarthkitSource
        }
        self.source_name = source_name

    def get_class (self):
        return self.class_registry.get(self.source_name)

class BaseSource (ABC):
    def __init__ (
        self,
        data_selection: DataSelection
    ):
        self.data_selection = data_selection
        self.file_paths = None
        self.ds = None

    @abstractmethod
    def _get_data (self) -> xr.DataArray:
        """
        Get data for the given data selection. Implement in subclasses.
        """
        pass

    def load (self) -> xr.DataArray:
        """Get data only if it hasn't been done yet"""
        if not self.ds:
            print("Get data...")
            t0 = time.time()
            self.ds = self._get_data()
            print(f"Dataset loading time: {time.time() - t0:.2f}s")
            # t0 = time.time()
            # self.ds = self._get_data_processpool(max_workers=36)
            # print(f"processpool time: {time.time() - t0:.2f}s")
        return self.ds

    def reload (self) -> xr.DataArray:
        """Force data reload"""
        self.ds = self._get_data()
        return self.ds

    def save (self, filepath: str | Path):
        """Save dataset in Zarr format in filepathc"""
        if not self.ds:
            self.ds = self.load()
        store = Path(filepath).with_suffix(".zarr")
        print(f"Saving dataset to {store}")
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        encoding_zarr = ({
            v: {"compressors": compressor} for v in self.ds.variables
        })
        self.ds.to_zarr(store, encoding=encoding_zarr, mode='w', consolidated=False)

class LocalSource (BaseSource):
    def __init__ (
        self,
        data_selection: DataSelection,
        root_path: str,
    ):
        super().__init__ (data_selection)
        self.path = Path(root_path)
        self.date_range = xr.date_range(
            start=self.data_selection.period.start,
            end=self.data_selection.period.end,
            freq=self.data_selection.period.freq
        )

    @staticmethod
    def _preprocess(ds: xr.Dataset, data: DataSelection, var_name: str = None) -> xr.DataArray:
        """
        Preprocess an xarray.Dataset according to CF conventions and a DataSelection.

        Features:
        - Ensures 'time' dimension exists and has proper attributes.
        - Selects spatial region (lon/lat) robustly:
            • Handles both 0–360 and -180–180 longitude systems.
            • Supports crossing the Greenwich meridian (e.g., lon=[350, 10]).
        - Selects vertical level(s) if available.
        - Returns a CF-aware DataArray ready for ML processing.
        """
        import cf_xarray  # ensures .cf accessor is registered on Dask workers
        import numpy as np
        import xarray as xr

        def _normalize_bounds(bounds):
            """Safely slice bounds"""
            if bounds is None:
                return slice(None)
            if isinstance(bounds, (int, float)):
                return bounds
            return slice(*bounds)

        time_dim = ds.cf["time"].name
        lon_dim = ds.cf["longitude"].name
        lat_dim = ds.cf["latitude"].name
        # Handle zero-time-dimension dataset
        if time_dim not in ds.dims:
            ds = ds.expand_dims(**{time_dim: [ds[time_dim].values]})
            ds = ds.assign_coords(
                **{time_dim: ds[time_dim].assign_attrs(standard_name="time", axis="T")}
            )
        # Vertical level selection
        level_sel_d = {}
        levhpa = getattr(data.variable, "levhpa", None)
        level_dim = None
        if 'vertical' in ds.cf.coordinates.keys() and levhpa is not None:
            level_dim = ds.cf["vertical"].name
            level_sel_d[level_dim] = _normalize_bounds(levhpa)
        # Spatial selection (lon/lat)
        lon = np.array(data.region.lon)
        lat = np.array(data.region.lat)
        lon_vals = ds[lon_dim].values
        # Normalize longitude convention
        if lon_vals.min() >= 0 and lon.min() < 0:
            # Dataset uses 0–360, region is -180–180
            lon = (lon + 360) % 360
        elif lon_vals.min() < 0 and lon.max() > 180:
            # Dataset uses -180–180, region is 0–360
            lon = ((lon + 180) % 360) - 180
        # Build selection dict
        selection_d = {
            lon_dim: _normalize_bounds(lon),
            lat_dim: _normalize_bounds(lat),
        } | level_sel_d
        # Select data
        if isinstance(data.variable, list):
            da = ds[var_name]
        else:
            da = ds[data.variable.name]
        # Handle longitude wrap-around (e.g., 350°–10° or -10°-40°)
        if lon[0] > lon[1]:
            da1 = da.sel(**{lon_dim: slice(lon[0], 360)})
            da2 = da.sel(**{lon_dim: slice(0, lon[1])})
            da = xr.concat([da1, da2], dim=lon_dim)
            # Apply remaining selections (lat, level, etc.)
            selection_d.pop(lon_dim, None)
            if selection_d:
                da = da.sel(**selection_d)
        else:
            da = da.sel(**selection_d)
        return da

    @abstractmethod
    def _get_data_filenames (self) -> dict:
        """
        Get the local data filenames for the given data selection. Implement in subclasses.
        """
        pass

class JunoLocalSource (LocalSource):
    """
    Collect Juno local data for the given data selection.
    """
    def __init__ (
        self,
        data_selection: DataSelection,
        root_path: str,
        engine: str,
        file_path_date_format: str,
        file_header: str,
        file_suffix: str,
        file_date_format: str,
        lead_time: timedelta,
        minus_timedelta: timedelta = None,
        plus_timedelta: timedelta = None
    ):
        super().__init__ (data_selection, root_path)
        self.engine = engine
        self.file_paths = self._get_data_filenames(
            file_path_date_format,
            file_header,
            file_suffix,
            file_date_format,
            lead_time,
            minus_timedelta,
            plus_timedelta
        )

    def _get_data_filenames(
        self,
        file_path_date_format: str,
        file_header: str,
        file_suffix: str,
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
            data_glob = f"{file_header}{date.strftime(file_date_format)}{forward_date.strftime(file_date_format)}{file_suffix}"

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

    def _get_data (self) -> xr.DataArray:
        # years = [str(date.year) for date in xr.date_range(start=data.period.start, end=data.period.end, freq='YS')]
        samples = list(self.file_paths['samples'].values())
        print(f"Samples: {len(samples)}, minus: {len(self.file_paths['minus_samples'])}, plus: {len(self.file_paths['plus_samples'])}, missed: {len(self.file_paths['missed_samples'])}")
        common_args = {
            "paths": samples,
            "combine": "by_coords",
            "coords": ["time"],
            "compat": "override" if (self.file_paths['minus_samples'] or self.file_paths['plus_samples']) else "no_conflicts",
            "engine": self.engine,
            "indexpath": "",
            "chunks": "auto",  # {}, {"time": 1}
            "parallel": True,
            "decode_timedelta": True,
            "backend_kwargs": {},
            "preprocess": partial(self._preprocess, data=self.data_selection),
            "decode_cf": True,
            "errors": "warn",
        }
        if isinstance(self.data_selection.variable, list):
            var_ds_list = []
            for var in self.data_selection.variable:
                if self.engine == "cfgrib":
                    common_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": var.name}} # not currently possible to filter with a list of keys (see https://github.com/ecmwf/cfgrib/issues/138)
                # TODO add support for other engines
                common_args["preprocess"] = partial(self._preprocess, data=self.data_selection, var_name=var.name)
                var_ds_list.append(xr.open_mfdataset(**common_args))
            return xr.merge(
                var_ds_list,
                compat="no_conflicts",
                combine_attrs="no_conflicts"
            )
        else:
            if self.engine == "cfgrib":
                common_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": self.data_selection.variable.name}}
            # TODO add support for other engines
            return xr.open_mfdataset(**common_args)

class EarthkitSource (BaseSource):
    """
    Collect data using ECMWF new earthkit library.
    """
    def __init__ (
        self,
        data_selection: DataSelection,
        provider: str,
        dataset: str,
        request_extra_args: dict = None,
        xarray_args: dict = None
    ):
        super().__init__ (data_selection)
        self.provider = provider
        self.dataset = dataset
        self.request_extra_args = request_extra_args
        self.xarray_args = xarray_args

    @staticmethod
    def _generate_hours (freq_str):
        value = int(freq_str[:-1])
        if freq_str[-1] != 'h':
            raise ValueError("Only 'h' (hours) frequency supported")

        times = []
        current = datetime.strptime("00:00", "%H:%M")
        while current.hour < 24:
            times.append(current.strftime("%H:%M"))
            current += timedelta(hours=value)
            if current.hour == 0:  # wrapped past midnight
                break
        return times

    def _get_data (self):
        import earthkit.data as ekd
        # samples = list(self.file_paths['samples'].values())
        var_name_list = [v.name for v in self.data_selection.variable] if isinstance(self.data_selection.variable, list) else [self.data_selection.variable.name]
        dates = f"{self.data_selection.period.start.strftime('%Y-%m-%d')}/{self.data_selection.period.end.strftime('%Y-%m-%d')}"
        time_freq = self._generate_hours(self.data_selection.period.freq)
        area = [
            self.data_selection.region.lat[0],
            self.data_selection.region.lon[0],
            self.data_selection.region.lat[1],
            self.data_selection.region.lon[1]
        ]
        print(f"Requesting {var_name_list} ({dates} {time_freq}) in region {area} from {self.provider}:{self.dataset}")
        request_d = dict(
            variable=var_name_list,
            area=area,
            date=dates,
            time=time_freq,
            **self.request_extra_args
        )
        # print(request_d)
        return ekd.from_source(
            self.provider,
            self.dataset,
            **request_d
        ).to_xarray(**self.xarray_args)
