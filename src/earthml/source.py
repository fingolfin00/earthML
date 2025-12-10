import certifi, os
# Ensure SSL and Requests use certifi CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import time
# from copy import deepcopy
from rich import print
# from rich.pretty import pprint
# from rich.table import Table
from abc import ABC, abstractmethod
import dask
from dask.distributed import wait
import cf_xarray
import xarray as xr
import pandas as pd
from pathlib import Path
from functools import partial
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
# from concurrent.futures import ProcessPoolExecutor
from functools import partial
from zarr.codecs import BloscCodec
import earthkit.data as ekd
# Local imports
from .dataclasses import DataSource, DataSelection, Sample
from .utils import generate_date_range, _guess_dim_name, get_lonlat_coords, get_ds_resolution, \
    generate_hours, subset_ds, regrid_to_rectilinear, print_ds_info

class SourceRegistry:
    def __init__ (self, source_name: str):
        self.class_registry = {
            "xarray-local": XarrayLocalSource,
            "juno-local": JunoLocalSource,
            "earthkit": EarthkitSource
        }
        self.source_name = source_name

    def get_class (self):
        return self.class_registry.get(self.source_name)

class BaseSource (ABC):
    def __init__ (
        self,
        datasource: DataSource
    ):
        self.data_selection = datasource.data_selection
        self.source_name = datasource.source
        self.date_range = generate_date_range(self.data_selection.period)
        print(f"Date range: length {len(self.date_range)}, {self.date_range[0]} to {self.date_range[-1]}")
        # print(self.date_range)
        self.elements = Sample()
        self.ds = None

    def __add__(self, other: "BaseSource") -> "BaseSource":
        if not isinstance(other, BaseSource):
            return NotImplemented
        return SumSource(self, other)

    def __radd__(self, other: "BaseSource") -> "BaseSource":
        # so sum([s1, s2, s3]) works
        if other == 0:
            return self
        return self.__add__(other)

    @abstractmethod
    def _get_data (self) -> xr.Dataset:
        """
        Get data for the given data selection. Implement in subclasses.
        """
        pass

    def load(self) -> xr.Dataset:
        """Get data only if it hasn't loaded yet"""
        if self.ds is None:
            print(f"Load data from {self.source_name}...")
            t0 = time.time()
            ds = self._get_data()

            # Persist here to materialise the dataset on the cluster
            # and shrink the graph that lives on the client.
            if hasattr(ds, "chunk"):  # i.e. Dask-backed
                ds = ds.chunk()   # ensure it’s dask-backed (no-op if already)
                ds = ds.persist()
                wait(ds) # block load() until materialised

            self.ds = ds
            print(f" → loading time: {time.time() - t0:.2f}s")
            print(f" → dataset shape: {self.ds.sizes}")
        return self.ds

    def reload (self) -> xr.Dataset:
        """Force data reload"""
        self.ds = self._get_data()
        return self.ds

    def save (self, filepath: str | Path):
        """Save dataset in Zarr format in filepath"""
        if not self.ds:
            self.ds = self.load()
        store = Path(filepath).with_suffix(".zarr")
        print(f"Saving dataset to {store}")
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        encoding_zarr = ({
            v: {"compressors": compressor} for v in self.ds.variables
        })
        self.ds.to_zarr(store, encoding=encoding_zarr, mode='w', consolidated=False)

class SumSource (BaseSource):
    def __init__(self, left: BaseSource, right: BaseSource):
        # Compatibility checks that don't require loading data
        # TODO add support for concat different regions
        if left.data_selection.region != right.data_selection.region:
            raise ValueError("Cannot add sources with different region")

        # Build a synthetic DataSource for the combined source
        combined_datasource = left.datasource + right.datasource
        super().__init__(combined_datasource)

        self._left = left
        self._right = right

    def _get_data(self) -> xr.Dataset:
        # This is where we finally touch the underlying data
        ds_left = self._left.load()
        ds_right = self._right.load()

        # Decide concat dimension
        time_dim = _guess_dim_name(
            ds_left, "time", ["valid_time", "time_counter"]
        )
        if time_dim is None:
            raise ValueError("Could not infer time dimension for concatenation")

        # Intersect data variables
        left_vars = set(ds_left.data_vars)
        right_vars = set(ds_right.data_vars)

        common_vars = sorted(left_vars & right_vars)
        if not common_vars:
            raise ValueError(
                "No common variables to concatenate between sources. "
                f"left={sorted(left_vars)}, right={sorted(right_vars)}"
            )

        extra_left = sorted(left_vars - right_vars)
        extra_right = sorted(right_vars - left_vars)
        if extra_left or extra_right:
            print(
                "[yellow]Warning:[/yellow] dropping non-common variables when adding sources:\n"
                f"  only in left:  {extra_left}\n"
                f"  only in right: {extra_right}"
            )

        ds_left_sel = ds_left[common_vars]
        ds_right_sel = ds_right[common_vars]

        # Intersect coords (by name)
        left_coords = set(ds_left.coords)
        right_coords = set(ds_right.coords)

        common_coords = left_coords & right_coords
        only_left_coords = left_coords - right_coords
        only_right_coords = right_coords - left_coords

        if only_left_coords or only_right_coords:
            print(
                "[yellow]Warning:[/yellow] dropping non-common coordinates when adding sources:\n"
                f"  only in left:  {sorted(only_left_coords)}\n"
                f"  only in right: {sorted(only_right_coords)}"
            )

        # Drop coords that are not shared
        ds_left_sel = ds_left_sel.drop_vars(list(only_left_coords), errors="ignore")
        ds_right_sel = ds_right_sel.drop_vars(list(only_right_coords), errors="ignore")

        # Concatenate lazily (xarray + dask)
        ds_combined = xr.concat([ds_left, ds_right], dim=time_dim)

        return ds_combined

class XarrayLocalSource (BaseSource):
    def __init__ (
        self,
        datasource: DataSource,
        root_path: str | Path,
        xarray_args: dict = None
    ):
        super().__init__ (datasource)
        self.path = Path(root_path)
        self.elements.samples = self.date_range
        self.xarray_args = {} if xarray_args is None else xarray_args

    def _get_data (self) -> xr.Dataset:
        self.ds = xr.open_dataset(self.path, **self.xarray_args)
        return self.ds

class MFXarrayLocalSource (BaseSource):
    def __init__ (
        self,
        datasource: DataSource,
        root_path: str,
    ):
        super().__init__ (datasource)
        self.path = Path(root_path)

    def _preprocess(
        self,
        ds: xr.Dataset,
        data: DataSelection,
        var_name: str | None = None,
    ) -> xr.Dataset:
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
        import cf_xarray  # ensure .cf accessor on workers
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
                **{
                    time_dim: ds[time_dim].assign_attrs(
                        standard_name="time",
                        axis="T",
                    )
                }
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
    def _get_data_filenames (self) -> Sample:
        """
        Get the local data filenames for the given data selection. Implement in subclasses.
        """
        pass

class JunoLocalSource (MFXarrayLocalSource):
    """
    Collect Juno local data for the given data selection.
    """
    def __init__ (
        self,
        datasource: DataSource,
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
        super().__init__ (datasource, root_path)
        self.engine = engine
        self.elements = self._get_data_filenames(
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
    ) -> Sample:
        """Get the data filenames for the given data selection."""

        s = Sample(extra={"plus_samples": [], "minus_samples": []})
        for date in self.date_range:
            previous_date = date - lead_time
            data_path = self.path.joinpath(previous_date.strftime(file_path_date_format))
            data_glob = f"{file_header}{previous_date.strftime(file_date_format)}{date.strftime(file_date_format)}{file_suffix}"

            # files_exact = [p for p in data_path.glob(data_glob) if p.is_file()]
            files_exact = sorted(
                (p for p in data_path.glob(data_glob) if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True  # newest first
            )

            if len(files_exact) > 1:
                print(f"{date}: {len(files_exact)} matches, keeping {files_exact[:1]}")
                files_exact = files_exact[:1]  # keep latest only, still a list
            # print(f"{date}, path:", files_exact)

            # Try direct match first
            if files_exact:
                s.samples[date] = files_exact[0]
                continue

            # Fallback search
            found = False
            if minus_timedelta and plus_timedelta:
                # print(minus_timedelta, plus_timedelta)
                for delta, store in [
                    (-minus_timedelta, s.extra['minus_samples']),
                    (plus_timedelta, s.extra['plus_samples']),
                ]:
                    test_date = date + delta
                    test_glob = f"{file_header}{previous_date.strftime(file_date_format)}{test_date.strftime(file_date_format)}*"
                    test_files = [p for p in data_path.glob(test_glob) if p.is_file()]
                    # print(f"New file {delta}: {test_files}")
                    if test_files:
                        s.samples[date] = test_files[0]
                        # s.samples.extend(test_files)
                        store.append(date)
                        found = True
                        break

            if not found:
                print(f"Missed sample: {date}")
                s.missed.add(date)

        return s

    def _get_data (self) -> xr.Dataset:
        # years = [str(date.year) for date in xr.date_range(start=data.period.start, end=data.period.end, freq='YS', inclusive='left')]
        # print(f"{self.source_name} missed dates: {self.elements.missed}")
        samples = [s for date, s in self.elements.samples.items() if date not in self.elements.missed]
        print(f"Samples: {len(samples)}, minus: {len(self.elements.extra['minus_samples'])}, plus: {len(self.elements.extra['plus_samples'])}, missed: {len(self.elements.missed)}")
        common_args = {
            "paths": samples,
            "combine": "by_coords",
            "coords": ["time"],
            "compat": "override" if (self.elements.extra['minus_samples'] or self.elements.extra['plus_samples']) else "no_conflicts",
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
        datasource: DataSource,
        provider: str,
        dataset: str,
        split_request: bool = False,
        request_extra_args: dict = None,
        xarray_args: dict = None
    ):
        super().__init__ (datasource)
        self.elements.samples = self.date_range
        self.provider = provider
        self.dataset = dataset
        self.split_request = split_request
        self.request_extra_args = request_extra_args
        self.xarray_args = xarray_args

    def _get_data (self):
        # samples = list(self.elements['samples'].values())
        samples = [s for s in self.elements.samples if s not in self.elements.missed] # TODO refactor to BaseSource?
        print(f"Samples: {len(samples)}, missed: {len(self.elements.missed)}")
        var_name_list = [v.name for v in self.data_selection.variable] if isinstance(self.data_selection.variable, list) else [self.data_selection.variable.name]
        start = self.data_selection.period.start
        end = self.data_selection.period.end
        dates = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        area = [
            self.data_selection.region.lat[0],
            self.data_selection.region.lon[0],
            self.data_selection.region.lat[1],
            self.data_selection.region.lon[1]
        ]
        print(f"Requesting {var_name_list} ({dates} {time_freq}) in region {area} from {self.provider}:{self.dataset}")
        print(f"Check request status: https://cds.climate.copernicus.eu/requests?tab=all")
        years = xr.date_range(start=start, end=end, freq="YS")
        print(f"Requesting {var_name_list} ({dates}, {self.data_selection.period.freq}) in region {area} from {self.provider}:{self.dataset}")
        print(f"Check request status: https://cds.climate.copernicus.eu/requests?tab=all")
        # print(years)
        if self.split_request and end - start > pd.to_timedelta('365 days'):
            # Split into yearly chunks
            if years[0] > start:
                years = xr.date_range(start, start, periods=1).append(years)
            if years[-1] <= end: # last date needs to be one day ahead to compensate for inclusive end
                years = years.append(xr.date_range(end+timedelta(days=1), end+timedelta(days=1), periods=1))
            # print(years)
            datasets = []
            for y1, y2 in zip(years[:-1], years[1:]):
                y2 = y2 - timedelta(days=1)  # inclusive end
                print(f" → Fetching chunk: {y1:%Y-%m-%d} to {y2:%Y-%m-%d}")
                request_d = dict(
                    variable=var_name_list,
                    area=area,
                    date=f"{y1:%Y-%m-%d}/{y2:%Y-%m-%d}",
                    time=time_freq,
                    **self.request_extra_args,
                )
                ds_chunk = ekd.from_source(self.provider, self.dataset, **request_d).to_xarray(**self.xarray_args)
                time_dim = ds_chunk.cf['time'].name
                datasets.append(ds_chunk)
            # for i, ds in enumerate(datasets):
            #     print(f"\n==== Dataset {i}: VARIABLES ====")
            #     for v in ds.variables:
            #         print(v, ds[v].dims, ds[v].shape)
            # print(time_dim)
            # Combine all datasets
            ds_all = xr.concat(
                datasets,
                dim=time_dim,
                # coords='minimal',  # Only use coords that vary along concat dimension
                # compat='override',  # Override minor inconsistencies
                combine_attrs='override'  # Handle attribute conflicts
            )
        else:
            request_d = dict(
                variable=var_name_list,
                area=area,
                date=dates,
                time=time_freq,
                **self.request_extra_args,
            )
            ds_all = ekd.from_source(self.provider, self.dataset, **request_d).to_xarray(**self.xarray_args)
            time_dim = ds_all.cf['time'].name
        # for v in ds_all.variables:
        #     print(v, ds_all[v].dims, ds_all[v].shape)
        # Drop missing samples
        ds_all = ds_all.drop_sel({time_dim: list(self.elements.missed)})
        # for v in ds_all.variables:
        #     print(v, ds_all[v].dims, ds_all[v].shape)
        return ds_all
