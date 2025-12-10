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
        concat_dim: str = None,
    ):
        super().__init__ (datasource)
        self.path = Path(root_path)
        self.concat_dim = concat_dim

    def _preprocess(
        self,
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
            time_dim = _guess_dim_name(ds, leadtime.name)
        else:
            # Fall back to CF time / common time dim names
            time_dim = _guess_dim_name(ds, "time", ["valid_time", "time_counter"])

        if not time_dim:
            raise ValueError("Could not find a time dimension or CF time axis")

        # --- Ensure this dimension exists in ds.dims --------------------
        # Some files may have a scalar time coordinate: make it a length-1 dimension
        if time_dim not in ds.dims and time_dim in ds.coords:
            coord = ds[time_dim]
            ds = ds.expand_dims({time_dim: [coord.values]})
            ds = ds.assign_coords(
                **{
                    time_dim: ds[time_dim].assign_attrs(
                        standard_name="time",
                        axis="T",
                    )
                }
            )

        lon_coord, lat_coord = get_lonlat_coords(ds)
        # print("Lon coord:", lon_coord, ", lat coord:", lat_coord)
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
        minus_timedelta: relativedelta = None,
        plus_timedelta: relativedelta = None
    ) -> Sample:
        """Get the data filenames for the given data selection."""

        s = Sample(extra={"plus_samples": [], "minus_samples": []})
        for date in self.date_range:
            previous_date = date - lead_time
            data_path = self.path.joinpath(previous_date.strftime(file_path_date_format))
            if both_data_and_previous_date_in_file:
                data_glob = f"{file_header}{previous_date.strftime(file_date_format)}{file_suffix}"
            else:
                data_glob = f"{file_header}{previous_date.strftime(file_date_format)}{date.strftime(file_date_format)}{file_suffix}"
            # print(f"{date}, glob:", data_path / data_glob)

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
            "combine": "nested" if self.concat_dim else "by_coords",
            "concat_dim": self.concat_dim,
            "coords": "different", # ["time"],
            # if minus/plus_samples time coordinate stepping might be irregular so override
            "compat": "override" if (self.elements.extra['minus_samples'] or self.elements.extra['plus_samples']) else "no_conflicts",
            "engine": self.engine,
            "chunks": {"time": -1},
            # "chunks": "auto",  # {}, {"time": 1}
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
                    common_args["indexpath"] = ""
                # untested support for netcdf4
                common_args["preprocess"] = partial(self._preprocess, data=self.data_selection, var_name=var.name)
                var_ds_list.append(xr.open_mfdataset(**common_args))
            ds = subset_ds(self.data_selection, xr.merge(
                var_ds_list,
                compat="no_conflicts",
                combine_attrs="no_conflicts"
            ))
        else:
            if self.engine == "cfgrib":
                common_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": self.data_selection.variable.name}}
                common_args["indexpath"] = ""
            # Tested support for netcdf4
            ds = xr.open_mfdataset(**common_args)
            lat_res, lon_res = get_ds_resolution(ds)
            print(f"Native resolutions: lat {lat_res:.2f}, lon {lon_res:.2f}")
            ds = subset_ds(self.data_selection, ds)

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
        dataset: str,
        split_request: bool = False,
        select_area_after_request: bool = False,
        request_type: bool = "subseasonal",
        request_extra_args: dict = None,
        xarray_args: dict = None,
        xarray_concat_dim: str = None,
        xarray_concat_extra_args: dict = None,
        regrid_resolution=None,  # float or (lat_res, lon_res) in degrees
        regrid_vars=None,
    ):
        super().__init__ (datasource)
        self.elements.samples = self.date_range
        self.provider = provider
        self.dataset = dataset
        self.split_request = split_request
        self.select_area_after_request = select_area_after_request
        self.request_type = request_type
        self.request_extra_args = request_extra_args
        self.xarray_args = xarray_args
        self.xarray_concat_dim = xarray_concat_dim
        self.xarray_concat_extra_args = xarray_concat_extra_args
        self.regrid_resolution = regrid_resolution
        self.regrid_vars = regrid_vars

    def _get_data (self):
        # samples = list(self.elements['samples'].values())
        samples = [s for s in self.elements.samples if s not in self.elements.missed] # TODO refactor to BaseSource?
        print(f"Samples: {len(samples)}, missed: {len(self.elements.missed)}")
        var_longname_list = [v.longname for v in self.data_selection.variable] if isinstance(self.data_selection.variable, list) else [self.data_selection.variable.longname]
        start = self.data_selection.period.start
        end = self.data_selection.period.end
        dates = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
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
                if self.request_type == "subseasonal":
                    time_freq = generate_hours(self.data_selection.period.freq)
                    request_time_args = dict(
                        date=f"{y1:%Y-%m-%d}/{y2:%Y-%m-%d}",
                        time=time_freq,
                        )
                elif self.request_type == "seasonal":
                    time_freq = ['01', '02' , '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
                    y22 = y2-relativedelta(years=1) if y2.strftime("%Y") != y1.strftime("%Y") else y2
                    request_time_args = dict(
                        year=xr.date_range(start=y1, end=y22, freq="YS").strftime("%Y").tolist(),
                        month=time_freq,
                        )
                else:
                    raise ValueError(f"Unsupported earthkit request type {self.request_type}")
                print(f" → Fetching chunk: {y1:%Y-%m-%d} to {y2:%Y-%m-%d}")
                request_d = dict(
                    **request_args,
                    **request_time_args,
                    **self.request_extra_args,
                )
                # print(request_d)
                ds_chunk = ekd.from_source(self.provider, self.dataset, **request_d).to_xarray(**self.xarray_args)
                xarray_concat_dim = ds_chunk.cf['time'].name if not self.xarray_concat_dim else self.xarray_concat_dim
                datasets.append(ds_chunk)
            # Combine all datasets
            ds_all = xr.concat(
                datasets,
                dim=xarray_concat_dim,
                **self.xarray_concat_extra_args,
            )
        else:
            if self.request_type == "subseasonal":
                time_freq = generate_hours(self.data_selection.period.freq)
                request_time_args = dict(
                    date=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                    time=time_freq,
                    )
            elif self.request_type == "seasonal":
                time_freq = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] # TODO only monthly timeseries supported
                request_time_args = dict(
                    year=xr.date_range(start=start, end=end, freq="YS").strftime("%Y").tolist(),
                    month=time_freq,
                    )
            else:
                raise ValueError(f"Unsupported earthkit request type {self.request_type}")
            request_d = dict(
                **request_args,
                **request_time_args,
                **self.request_extra_args,
            )
            ds_all = ekd.from_source(self.provider, self.dataset, **request_d).to_xarray(**self.xarray_args)
            xarray_concat_dim = ds_all.cf['time'].name if not self.xarray_concat_dim else self.xarray_concat_dim

        # Drop missing samples
        if self.elements.missed:
            ds_all = ds_all.drop_sel({xarray_concat_dim: list(self.elements.missed)})

        # Select area if necessary
        if self.select_area_after_request:
            ds_all = subset_ds(self.data_selection, ds_all)

        # Grid resolution
        lat_res, lon_res = get_ds_resolution(ds_all)
        print(f"Native resolutions: lat {lat_res:.2f}, lon {lon_res:.2f}")
        # Regrid if required
        if self.regrid_resolution is not None:
            print(f"Regridding to rectilinear grid with resolution {self.regrid_resolution}")
            ds_all = regrid_to_rectilinear(
                src_ds=ds_all,
                region=self.data_selection.region,
                resolution=self.regrid_resolution,
                vars_to_regrid=self.regrid_vars,
            )
            lat_res_regrid, lon_res_regrid = get_ds_resolution(ds_all)
            print(f"Target rectilinear resolutions: lat {lat_res_regrid:.2f}, lon {lon_res_regrid:.2f}")

        return ds_all
