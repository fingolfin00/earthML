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
from dask.utils import SerializableLock
import numpy as np
import cf_xarray
import xarray as xr
import pandas as pd
from pathlib import Path
from functools import partial
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY
# from concurrent.futures import ProcessPoolExecutor
from functools import partial
from zarr.codecs import BloscCodec
import earthkit.data as ekd
# Local imports
from .dataclasses import DataSource, DataSelection, Sample
from .utils import generate_date_range, _guess_dim_name, _guess_coord_name, get_lonlat_coords, get_ds_resolution, \
    generate_hours, subset_ds, regrid_to_rectilinear, print_ds_info, quickplot

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
        self.datasource = datasource
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

    def save (self, filepath: str | Path, consolidated: bool = False):
        """Save dataset in Zarr format in filepath"""
        if not self.ds:
            self.ds = self.load()
        store = Path(filepath)
        print(f"Saving dataset to {store}")
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        encoding_zarr = ({
            v: {"compressors": compressor} for v in self.ds.variables
        })
        self.ds.to_zarr(store, encoding=encoding_zarr, mode='w', consolidated=consolidated)

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
        ds_combined = xr.concat([ds_left_sel, ds_right_sel], dim=time_dim)

        return ds_combined

class XarrayLocalSource (BaseSource):
    def __init__ (
        self,
        datasource: DataSource,
        root_path: str | Path,
        xarray_args: dict = None,
    ):
        super().__init__ (datasource)
        self.path = Path(root_path)
        self.elements.samples = self.date_range
        self.xarray_args = {} if xarray_args is None else xarray_args

    def _get_data (self) -> xr.Dataset:
        self.ds = xr.open_dataset(self.path, **self.xarray_args)
        if self.elements.missed:
            time_coord = _guess_coord_name(self.ds, "time", ["valid_time", "time_counter"])
            for var in self.ds:
                print(self.ds[var].shape)
            # self.ds[time_coord] = [d for d in self.date_range if d not in self.elements.missed]
            self.ds[time_coord] = self.date_range
            print(self.ds[time_coord])
            self.ds = self.ds.drop_sel({time_coord: list(self.elements.missed)})
        return self.ds

class MFXarrayLocalSource (BaseSource):
    def __init__ (
        self,
        datasource: DataSource,
        root_path: str,
        concat_dim: str = None,
    ):
        super().__init__ (datasource)
        self.path = Path(root_path)
        self.concat_dim = concat_dim

    @staticmethod
    def _preprocess(
        ds: xr.Dataset,
        data: DataSelection,
        var_name: str | None = None,
    ) -> xr.Dataset:
        """
        Lightweight per-file preprocess for open_mfdataset.

        Responsibilities:
        - optionally select a single variable
        - ensure a time-like dimension exists (for concatenation)
        - optionally select a specific leadtime if present
        """
        import cf_xarray  # ensure .cf accessor on workers
        import xarray as xr
        import pandas as pd

        # --- Leadtime handling -----------------------------------------
        # Use first variable's leadtime if `data.variable` is a list
        var0 = data.variable[0] if isinstance(data.variable, list) else data.variable
        leadtime = var0.leadtime

        # --- Guess time-like dimension name -----------------------------
        # If we have a leadtime axis, we often want to use that name directly
        if leadtime is not None:
            # e.g. leadtime.name == "leadtime"
            time_coord = _guess_coord_name(ds, leadtime.name)
        else:
            # Fall back to CF time / common time dim names
            time_coord = _guess_coord_name(ds, "time", ["valid_time", "time_counter"])

        if not time_coord:
            print("Could not find a time coord or CF time axis, try to assign it")

        # print("time coord", time_coord)
        # --- Ensure this dimension exists in ds.dims --------------------
        # Some files may have a scalar time coordinate: make it a length-1 dimension
        if time_coord not in ds.dims and time_coord in ds.coords:
            coord = ds[time_coord]
            ds = ds.expand_dims({time_coord: [coord.values]})
            ds = ds.assign_coords(
                **{
                    time_coord: ds[time_coord].assign_attrs(
                        standard_name="time",
                        axis="T",
                    )
                }
            )

        lon_coord, lat_coord = get_lonlat_coords(ds)
        # print("Time coord", time_coord, "Lon coord:", lon_coord, ", lat coord:", lat_coord)
        # if ds[lon_coord].ndim > 1:
        #     print(ds[lon_coord].values)

        # --- Select variable --------------------------------------------
        if var_name is not None:
            da = ds[var_name]
        else:
            if isinstance(data.variable, list):
                da = ds[var0.name]  # use first variable's name
            else:
                da = ds[var0.name]

        # --- Select leadtime if present ---------------------------------
        if leadtime is not None:
            # Build target timedelta from value + unit (e.g. "3 days", "12 hours")
            td = pd.to_timedelta(f"{leadtime.value} {leadtime.unit}")

            # Cast to same dtype as coord (usually timedelta64[ns])
            coord_dtype = ds[leadtime.name].dtype
            target = td.to_numpy().astype(coord_dtype)

            da = da.sel({leadtime.name: target}, method="nearest")

        # We need to return a xr.Dataset otherwise xr.open_mfdataset with combine="nested"
        # returns a DataArray, which would break consistency further ahead
        return xr.Dataset(
            {da.name or var_name or var0.name: da},
            {lon_coord: ds[lon_coord], lat_coord: ds[lat_coord]} if lon_coord and lat_coord else None
        )

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
        lead_time: relativedelta,
        both_data_and_previous_date_in_file: bool = True,
        realizations: int | str = 1,
        minus_timedelta: timedelta = None,
        plus_timedelta: timedelta = None,
        concat_dim: str = None,
        regrid_resolution=None,  # float or (lat_res, lon_res) in degrees
        regrid_vars=None,
    ):
        super().__init__ (datasource, root_path, concat_dim)
        self.engine = engine
        self.elements = self._get_data_filenames(
            file_path_date_format,
            file_header,
            file_suffix,
            file_date_format,
            lead_time,
            both_data_and_previous_date_in_file,
            realizations,
            minus_timedelta,
            plus_timedelta
        )
        self.regrid_resolution = regrid_resolution
        self.regrid_vars = regrid_vars

    def _get_data_filenames(
        self,
        file_path_date_format: str,
        file_header: str,
        file_suffix: str,
        file_date_format: str,
        lead_time: relativedelta,
        both_data_and_previous_date_in_file: bool = True,
        realizations: int | str = 1,
        minus_timedelta: relativedelta = None,
        plus_timedelta: relativedelta = None
    ) -> Sample:
        """Get the data filenames for the given data selection."""

        s = Sample(extra={"plus_samples": [], "minus_samples": []})
        assert realizations == 'all' or realizations > 0
        for date in self.date_range:
            previous_date = date - lead_time
            data_path = self.path.joinpath(previous_date.strftime(file_path_date_format))
            if both_data_and_previous_date_in_file:
                data_glob = f"{file_header}{previous_date.strftime(file_date_format)}{date.strftime(file_date_format)}{file_suffix}"
            else:
                data_glob = f"{file_header}{previous_date.strftime(file_date_format)}{file_suffix}"
            # print(f"{date}, glob:", data_path / data_glob)

            # files_exact = [p for p in data_path.glob(data_glob) if p.is_file()]
            files_exact = sorted(
                (p for p in data_path.glob(data_glob) if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True  # newest first
            )

            if len(files_exact) > 1:
                if realizations == 'all':
                    r = len(files_exact)
                else:
                    r = realizations
                print(f"{date}: {len(files_exact)} matches")
                # print(f"{date}: {len(files_exact)} matches, keeping {files_exact[:r]}")
                files_exact = files_exact[:r]  # keep latest only, still a list
            # print(f"{date}, path:", files_exact)

            # Try direct match first
            if files_exact:
                s.samples[date] = files_exact
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
                    if both_data_and_previous_date_in_file:
                        test_glob = f"{file_header}{previous_date.strftime(file_date_format)}{test_date.strftime(file_date_format)}{file_suffix}"
                    else:
                        test_glob = f"{file_header}{previous_date.strftime(file_date_format)}{file_suffix}" # TODO look better into this
                    test_files = [p for p in data_path.glob(test_glob) if p.is_file()]
                    # print(f"New file {delta}: {test_files}")
                    if test_files:
                        if realizations == 'all':
                            r = len(test_files)
                        else:
                            r = realizations
                        s.samples[date] = test_files[:r]
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
        samples = [s for date, s in self.elements.samples.items() if date not in self.elements.missed] # list of lists
        dates = [date for date in self.elements.samples.keys() if date not in self.elements.missed]
        print(f"Samples: {len(samples)}, minus: {len(self.elements.extra['minus_samples'])}, plus: {len(self.elements.extra['plus_samples'])}, missed: {len(self.elements.missed)}")
        samples_d = {}
        lock = SerializableLock()
        if not hasattr(self, "concat_dim"): # backward compat
            self.concat_dim = None
        common_args = {
            "combine": "nested" if self.concat_dim else "by_coords",
            "concat_dim": "realization",
            "coords": "minimal" if (self.elements.extra['minus_samples'] or self.elements.extra['plus_samples']) else "different", # ["time"],
            # if minus/plus_samples time coordinate stepping might be irregular so override
            "compat": "override" if (self.elements.extra['minus_samples'] or self.elements.extra['plus_samples']) else "no_conflicts",
            "engine": self.engine,
            # "chunks": {'time': -1},
            "chunks": "auto",
            "parallel": True,
            "decode_timedelta": True,
            "backend_kwargs": {},
            "preprocess": partial(self._preprocess, data=self.data_selection),
            "decode_cf": True,
            "errors": "warn",
            "lock": lock,
        }
         # backward compat for old experiments without realization support
        if not isinstance(samples[0], list):
            # print(samples)
            print("Engine:", self.engine)
            common_args['paths'] = samples
            common_args["concat_dim"] = self.concat_dim
            common_args["decode_times"] = False
            if isinstance(self.data_selection.variable, list):
                var_ds_list = []
                for var in self.data_selection.variable:
                    if self.engine == "cfgrib":
                        common_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": var.name}} # not currently possible to filter with a list of keys (see https://github.com/ecmwf/cfgrib/issues/138)
                        common_args["indexpath"] = ""
                    # untested support for netcdf4
                    common_args["preprocess"] = partial(self._preprocess, data=self.data_selection, var_name=var.name)
                    var_ds_list.append(xr.open_mfdataset(**common_args))
                # lat_res, lon_res = get_ds_resolution(var_ds_list[0])
                # print(f"Native resolutions ({self.data_selection.variable[0]}): lat {lat_res:.2f}, lon {lon_res:.2f}")
                ds = subset_ds(self.data_selection, xr.merge(
                    var_ds_list,
                    compat="no_conflicts",
                    combine_attrs="no_conflicts"
                ))
                # Load time coord
                if self.concat_dim in ds.coords:
                    ds = ds.assign_coords({self.concat_dim: ds[self.concat_dim].load()})
            else:
                if self.engine == "cfgrib":
                    common_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": self.data_selection.variable.name}}
                    common_args["indexpath"] = ""
                # Tested support for netcdf4
                ds = xr.open_mfdataset(**common_args)
                # lat_res, lon_res = get_ds_resolution(ds_sample)
                # print(f"Native resolutions: lat {lat_res:.2f}, lon {lon_res:.2f}")
                ds = subset_ds(self.data_selection, ds)
                # Load time coord
                if self.concat_dim in ds_sample.coords:
                    ds = ds.assign_coords({self.concat_dim: ds[self.concat_dim].load()})

        else:
            for sample, date in zip(samples, dates):
                assert isinstance(sample, list), f"Sample should be a list but it is {type(sample)}"
                # print(sample)
                common_args['paths'] = sample
                common_args["concat_dim"] = "realization"
                if isinstance(self.data_selection.variable, list):
                    var_ds_list = []
                    for var in self.data_selection.variable:
                        if self.engine == "cfgrib":
                            common_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": var.name}} # not currently possible to filter with a list of keys (see https://github.com/ecmwf/cfgrib/issues/138)
                            common_args["indexpath"] = ""
                        # untested support for netcdf4
                        common_args["preprocess"] = partial(self._preprocess, data=self.data_selection, var_name=var.name)
                        var_ds_list.append(xr.open_mfdataset(**common_args))
                    # lat_res, lon_res = get_ds_resolution(var_ds_list[0])
                    # print(f"Native resolutions ({self.data_selection.variable[0]}): lat {lat_res:.2f}, lon {lon_res:.2f}")
                    ds_sample = subset_ds(self.data_selection, xr.merge(
                        var_ds_list,
                        compat="no_conflicts",
                        combine_attrs="no_conflicts"
                    ))
                    # Load time coord
                    if self.concat_dim in ds_sample.coords:
                        ds_sample = ds_sample.assign_coords({self.concat_dim: ds_sample[self.concat_dim].load()})
                else:
                    if self.engine == "cfgrib":
                        common_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": self.data_selection.variable.name}}
                        common_args["indexpath"] = ""
                    # Tested support for netcdf4
                    ds_sample = xr.open_mfdataset(**common_args)
                    # lat_res, lon_res = get_ds_resolution(ds_sample)
                    # print(f"Native resolutions: lat {lat_res:.2f}, lon {lon_res:.2f}")
                    ds_sample = subset_ds(self.data_selection, ds_sample)
                    # Load time coord
                    if self.concat_dim in ds_sample.coords:
                        ds_sample = ds_sample.assign_coords({self.concat_dim: ds_sample[self.concat_dim].load()})

                samples_d[date] = ds_sample

            # Combine realizations
            samples_len = [ds.sizes.get("realization", 1) for ds in samples_d.values()]
            # print(f"Samples length: {samples_len}")
            min_R = min(samples_len)
            for date, ds in samples_d.items():
                if "realization" in ds.dims:
                    dsR = ds.isel(realization=slice(0, min_R))
                    # Wipe realization coord info
                    R = dsR.sizes["realization"]
                    samples_d[date] = dsR.assign_coords(realization=np.arange(R))
            # for d in list(sorted(samples_d))[:3]:
            #     ds = samples_d[d]
            #     print(d, ds.sizes.get("realization"), ds.coords.get("realization").values, ds.coords["realization"].dtype)
            times = np.array(sorted(samples_d.keys()), dtype="datetime64[ns]")
            objs = [samples_d[d] for d in sorted(samples_d)]
            ds = xr.concat(
                objs=objs,
                dim=xr.IndexVariable(self.concat_dim, times) if self.concat_dim in ('time', 'valid_time', 'time_counter') else self.concat_dim,
                compat="broadcast_equals",
                join='exact',
                combine_attrs='drop_conflicts'
            )

        lat_res, lon_res = get_ds_resolution(ds)
        print(f"Horizontal resolutions: lat {lat_res:.2f}, lon {lon_res:.2f}")

        # Regrid if required
        if self.regrid_resolution is not None:
            print(f"Regridding to rectilinear grid with resolution {self.regrid_resolution}")
            ds = regrid_to_rectilinear(
                src_ds=ds,
                region=self.data_selection.region,
                resolution=self.regrid_resolution,
                vars_to_regrid=self.regrid_vars,
            )

            lat_res_regrid, lon_res_regrid = get_ds_resolution(ds)
            print(f"Target rectilinear resolutions: lat {lat_res_regrid:.2f}, lon {lon_res_regrid:.2f}")

        return ds

class EarthkitSource (BaseSource):
    """
    Collect data using ECMWF new earthkit library.
    """
    def __init__ (
        self,
        datasource: DataSource,
        provider: str,
        dataset: str = None,
        split_request: bool = False,
        split_month: int = 12,
        split_month_jump: list = None,
        select_area_after_request: bool = False,
        request_type: str = "subseasonal",
        request_extra_args: dict = None,
        to_xarray_args: dict = None,
        xarray_concat_dim: str = None,
        xarray_concat_extra_args: dict = None,
        regrid_resolution=None,  # float or (lat_res, lon_res) in degrees
        regrid_vars=None,
        earthkit_cache_dir=Path("/tmp/earthkit-cache/"),
    ):
        super().__init__ (datasource)

        ekd.config.set("cache-policy", "user")
        ekd.config.set("user-cache-directory", earthkit_cache_dir)

        self.elements.samples = self.date_range
        self.provider = provider
        self.dataset = dataset
        self.split_request = split_request
        self.split_month = split_month
        self.split_month_jump = split_month_jump if split_month_jump else []
        self.select_area_after_request = select_area_after_request
        self.request_type = request_type
        self.request_extra_args = request_extra_args
        self.to_xarray_args = to_xarray_args
        self.xarray_concat_dim = xarray_concat_dim
        self.xarray_concat_extra_args = xarray_concat_extra_args
        self.regrid_resolution = regrid_resolution
        self.var_name_list = [v.name for v in self.data_selection.variable] if isinstance(self.data_selection.variable, list) else [self.data_selection.variable.name]
        self.regrid_vars = regrid_vars if regrid_vars is not None else self.var_name_list

        self._create_leadtime_dict()
        self._populate_missed()

    def _populate_missed (self):
        """Populate missed if some months are skipped for seasonal requests"""
        if self.request_type == "seasonal":
            start = self.data_selection.period.start
            end = self.data_selection.period.end
            skip_months = set(self.split_month_jump)

            self.elements.missed = {
                dt for dt in rrule(MONTHLY, dtstart=start, until=end) # TODO we only support monthly seasonal datasets
                if f"{dt.month:02d}" in skip_months
            }

    def _create_leadtime_dict (self):
        vars_ = (
            self.data_selection.variable
            if isinstance(self.data_selection.variable, list)
            else [self.data_selection.variable]
        )

        leadtime_pairs = []

        for v in vars_:
            lt = getattr(v, "leadtime", None)
            if lt is None:
                continue

            # resolve timedelta
            if hasattr(lt, "value") and hasattr(lt, "unit"):
                td = pd.to_timedelta(f"{lt.value} {lt.unit}")
            else:
                td = pd.to_timedelta(lt)

            name = getattr(lt, "name", "leadtime")
            leadtime_pairs.append((name, td))

        if not leadtime_pairs:
            self.leadtime_d = {}
        else:
            names = {n for n, _ in leadtime_pairs}
            times = {t for _, t in leadtime_pairs}

            if len(names) > 1 or len(times) > 1:
                raise ValueError(
                    f"Leadtime inconsistent across variables: "
                    f"names={sorted(names)}, leadtimes={sorted(times)}"
                )

            name, td = leadtime_pairs[0]
            self.leadtime_d = {name: td}

    def _get_data (self):
        # samples = list(self.elements['samples'].values())
        samples = [s for s in self.elements.samples if s not in self.elements.missed] # TODO refactor to BaseSource?
        print(f"Samples: {len(samples)}, missed: {len(self.elements.missed)}")
        var_longname_list = [v.longname for v in self.data_selection.variable] if isinstance(self.data_selection.variable, list) else [self.data_selection.variable.longname]
        start = self.data_selection.period.start
        end = self.data_selection.period.end
        dates = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        all_months = ['01', '02' , '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        months_splitted = [
            [m for m in all_months[i:i+self.split_month] if m not in self.split_month_jump]
            for i in range(0, len(all_months), self.split_month)
        ]
        # print(f"Months requested: {months_splitted}")
        # Convert singletons to strings and clean up empty/None elements
        months_splitted = [chunk[0] if len(chunk) == 1 else chunk for chunk in months_splitted]
        months_splitted = [x for x in months_splitted if x]
        # print(f"Months requested cleaned-up: {months_splitted}")
        area = [
            self.data_selection.region.lat[0],
            self.data_selection.region.lon[0],
            self.data_selection.region.lat[1],
            self.data_selection.region.lon[1]
        ]
        if self.select_area_after_request:
            request_args = dict(variable=var_longname_list)
        else:
            request_args = dict(variable=var_longname_list, area=area)
        years = xr.date_range(start=start, end=end, freq="YS")
        print(f"Requesting {var_longname_list} ({dates}, {self.data_selection.period.freq}) in region {area} from {self.provider}:{self.dataset}")
        print(f"Check request status: https://cds.climate.copernicus.eu/requests?tab=all")
        # print(years)

        def _fetch_chunks (request_time_args_list, start, end, request_other_args):
            """Helper to fetch chunked datasets using ekd"""
            ds_chunks = []
            for req_time_arg in request_time_args_list:
                months_req = ""
                if 'month' in req_time_arg:
                    months_req = req_time_arg['month'] if isinstance(req_time_arg['month'], list) else [req_time_arg['month']]
                    n_months_req = len(months_req)
                print(f" → Fetching chunk: {start:%Y-%m-%d} to {end:%Y-%m-%d} {months_req}")
                request_d = dict(
                    **request_other_args,
                    **req_time_arg,
                )
                if self.dataset:
                    ds_chunk = ekd.from_source(self.provider, self.dataset, **request_d).to_xarray(**self.to_xarray_args)
                else:
                    ds_chunk = ekd.from_source(self.provider, **request_d).to_xarray(**self.to_xarray_args)
                print(f"   Chunk size: {ds_chunk.sizes}")
                # print(f"   Chunk coords: {ds_chunk.coords}")

                if self.leadtime_d:
                    td = next(iter(self.leadtime_d.values()))
                    leadtime_name = next(iter(self.leadtime_d.keys()))
                    # TODO we currently support only the case with 'time' coord to infer realizations
                    if "leadtime" in ds_chunk.coords and 'time' in ds_chunk.coords and td is not None:
                        print(f"   Using leadtime {self.leadtime_d}")
                        lt_values = ds_chunk[leadtime_name].values
                        unique_lt = np.unique(lt_values)
                        time_values = ds_chunk['time'].values
                        unique_time = np.unique(time_values)
                        n_unique_time = len(unique_time)
                        n_leadtime = n_unique_time - n_months_req + 1
                        print(f"   leadtimes detected: {n_leadtime}")
                        n_realizations = len(lt_values) / n_months_req / n_leadtime
                        assert n_realizations.is_integer(), f"Number of realizations cannot be computed, check the dataset"
                        n_realizations = int(n_realizations)
                        print(f"   realizations detected: {n_realizations}")
                        # coord_u = np.unique(coord)
                        # Cast to same dtype as coord (usually timedelta64[ns])
                        coord_dtype = ds_chunk[leadtime_name].dtype
                        target = td.to_numpy().astype(coord_dtype)
                        # print(f"   requested leadtime {target}")
                        nearest_lt = unique_lt[np.argmin(np.abs(unique_lt - target))]
                        print(f"   selected leadtime {pd.Timedelta(nearest_lt)}")
                        mask = (lt_values == nearest_lt)
                        # print(f"   leadtime sel mask: {mask}")
                        # print(f"   total leadtimes: {lt_values}")
                        ds_sel = ds_chunk.isel({leadtime_name: mask})
                        # print(f"   Size after leadtime sel: {ds_sel.sizes}")
                        n_sel_lt = len(ds_sel['leadtime'].values)
                        n_sel_re = n_sel_lt / n_months_req
                        assert n_sel_re.is_integer() and int(n_sel_re) == n_realizations, f"Number of realizations not coherent, try single-month requests"
                        # print(f"   Coords after leadtime sel: {ds_sel.coords}")
                        re_values = ds_sel['leadtime'].values
                        # unique_re = np.unique(re_values)
                        # print(f"   Total leadtimes: {len(lt_values)}, times: {len(time_values)}, realizations: {re_values}")
                        # print(f"   Unique leadtimes: {len(unique_lt)}, times: {len(unique_time)}, realizations: {unique_re}")
                        # print(f"   Uniques leadtimes: {unique_lt}")
                        # print(f"   Uniques times: {unique_time}")
                        # print(unique_lt)
                        # print(unique_re)
                        if "realization" in ds_sel.coords and "realization" not in ds_sel.dims:
                        # keep old realization as metadata
                            ds_sel = ds_sel.assign_attrs(source_realization=str(ds_sel["realization"].values))
                            ds_sel = ds_sel.drop_vars("realization")
                        ds_sel = ds_sel.rename({leadtime_name: "realization"})
                        ds_sel = ds_sel.assign_coords(realization=np.arange(len(re_values)))
                        ds_sel = ds_sel.assign_coords({leadtime_name: nearest_lt})
                        if "time" in ds_sel.coords and "realization" in ds_sel["time"].dims:
                            t = ds_sel["time"].values
                            if np.all(t == t[0]):
                                ds_sel = ds_sel.assign_coords(time=t[0])
                            else:
                                # if not identical, pick one (or decide a rule)
                                ds_sel = ds_sel.assign_coords(time=t[0])
                        # make time a 1-length dimension for concat
                        ds_chunk = ds_sel.expand_dims(time=[ds_sel["time"].item()])
                        print(f"   Size after all processing: {ds_chunk.sizes}")
                xarray_concat_dim = ds_chunk.cf['time'].name if not self.xarray_concat_dim else self.xarray_concat_dim
                # print(xarray_concat_dim)
                ds_chunks.append(ds_chunk)
            return xarray_concat_dim, ds_chunks

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
                request_time_args_list = []
                if self.request_type == "subseasonal":
                    time_freq = generate_hours(self.data_selection.period.freq)
                    request_time_args = dict(
                        date=f"{y1:%Y-%m-%d}/{y2:%Y-%m-%d}",
                        time=time_freq,
                    )
                    request_time_args_list.append(request_time_args)
                elif self.request_type == "seasonal":
                    y22 = y2-relativedelta(years=1) if y2.strftime("%Y") != y1.strftime("%Y") else y2
                    for m in months_splitted:
                        request_time_args = dict(
                            year=xr.date_range(start=y1, end=y22, freq="YS").strftime("%Y").tolist(),
                        )
                        if "month" not in self.request_extra_args:
                            request_time_args['month'] = m
                        request_time_args_list.append(request_time_args)
                else:
                    raise ValueError(f"Unsupported earthkit request type {self.request_type}")
                xarray_concat_dim, ds_chunks = _fetch_chunks(request_time_args_list, y1, y2, request_args | self.request_extra_args)
                datasets.extend(ds_chunks)

            # Combine all datasets
            ds_all = xr.concat(
                datasets,
                dim=xarray_concat_dim,
                **self.xarray_concat_extra_args,
            )
        else:
            request_time_args_list = []
            if self.request_type == "subseasonal":
                time_freq = generate_hours(self.data_selection.period.freq)
                request_time_args = dict(
                    date=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                    time=time_freq,
                )
                request_time_args_list.append(request_time_args)
            elif self.request_type == "seasonal":
                for m in months_splitted:
                    request_time_args = dict(
                        year=xr.date_range(start=start, end=end, freq="YS").strftime("%Y").tolist(),
                    )
                    if "month" not in self.request_extra_args:
                        request_time_args['month'] = m
                    request_time_args_list.append(request_time_args)
            else:
                raise ValueError(f"Unsupported earthkit request type {self.request_type}")
            xarray_concat_dim, datasets = _fetch_chunks(request_time_args_list, start, end, request_args | self.request_extra_args)
            # Combine all datasets
            ds_all = xr.concat(
                datasets,
                dim=xarray_concat_dim,
                **self.xarray_concat_extra_args,
            )

        # Drop unused variables
        ds_all = ds_all.drop_vars([v for v in ds_all.data_vars if v not in self.var_name_list])
        # Drop missing samples
        xarray_concat_dim = ds_all.cf['time'].name if not self.xarray_concat_dim else self.xarray_concat_dim
        if self.elements.missed:
            ds_all = ds_all.drop_sel({xarray_concat_dim: list(self.elements.missed)}, errors='ignore')

        # Select area if necessary
        if self.select_area_after_request:
            ds_all = subset_ds(self.data_selection, ds_all)

        # Grid resolution # TODO maybe refactor to BaseSource
        lat_res, lon_res = get_ds_resolution(ds_all)
        print(f"Native resolutions: lat {lat_res:.2f}, lon {lon_res:.2f}")
        # Regrid if required
        if self.regrid_resolution is not None:
            print(f"Regridding {self.regrid_vars} to rectilinear grid with resolution {self.regrid_resolution}")
            ds_all = regrid_to_rectilinear(
                src_ds=ds_all,
                region=self.data_selection.region,
                resolution=self.regrid_resolution,
                vars_to_regrid=self.regrid_vars,
            )
            lat_res_regrid, lon_res_regrid = get_ds_resolution(ds_all)
            print(f"Target rectilinear resolutions: lat {lat_res_regrid:.2f}, lon {lon_res_regrid:.2f}")
        return ds_all
