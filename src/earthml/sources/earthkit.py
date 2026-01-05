from pathlib import Path
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY
import tempfile, os, shutil
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import pandas as pd
import xarray as xr
import earthkit.data as ekd
from rich import print

from ..dataclasses import DataSource
from ..utils import retry_fetch_after_hdf_err, generate_hours, get_ds_resolution, subset_ds, regrid_to_rectilinear, _guess_dim_name
from .base import BaseSource

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
        request_type: str = "hourly", # on of hourly, daily, monthly
        request_extra_args: dict = None,
        to_xarray_args: dict = None,
        xarray_concat_dim: str = None,
        xarray_concat_extra_args: dict = None,
        regrid_resolution: float | tuple[float, float] = None,  # float or (lat_res, lon_res) in degrees
        regrid_vars: list[str] = None,
        earthkit_cache_dir: str = Path("/tmp/earthkit-cache/"),
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
        self.ekd_version = ekd.__version__

        self._create_leadtime_dict()
        self._populate_missed()

        # DEBUG
        # import os, certifi
        # print("[cyan]SSL_CERT_FILE[/cyan] =", os.environ.get("SSL_CERT_FILE"))
        # print("[cyan]REQUESTS_CA_BUNDLE[/cyan] =", os.environ.get("REQUESTS_CA_BUNDLE"))
        # print("[cyan]certifi.where()[/cyan] =", certifi.where())

    def _populate_missed (self):
        """Populate missed if some months are skipped for seasonal requests"""
        if self.request_type == "monthly":
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
        print(f"Requesting {var_longname_list} ({dates}, {self.data_selection.period.freq}) in region {area} from {self.provider}:{self.dataset} (ekd_ver={self.ekd_version})")
        print(f"Check request status: https://cds.climate.copernicus.eu/requests?tab=all")
        # print(years)

        def _earthkit_source_path (src) -> Path | None:
            """
            Try to get a real filesystem path from an earthkit source.
            Works for file-backed sources.
            """
            for attr in ("path", "path_or_url", "url"):
                p = getattr(src, attr, None)
                if isinstance(p, str) and p.startswith("/"):
                    return Path(p)
            # some earthkit objects expose a list of files/parts
            parts = getattr(src, "parts", None)
            if parts:
                for part in parts:
                    p = getattr(part, "path", None) or getattr(part, "path_or_url", None)
                    if isinstance(p, str) and p.startswith("/"):
                        return Path(p)
            return None

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
                # print(request_d)

                if self.dataset:
                    src_ekd_params = {"name": self.provider, "dataset": self.dataset} | request_d
                    # src_ekd = ekd.from_source(self.provider, self.dataset, **request_d)
                else:
                    src_ekd_params = {"name": self.provider} | request_d

                # print(src_ekd_params)
                def _fetch_ekd_src ():
                    return ekd.from_source(**src_ekd_params)
                src_ekd = retry_fetch_after_hdf_err(_fetch_ekd_src, error_re=r"NetCDF:.*HDF error", base_sleep=5, tries=2, delete_bad_file=True, delete_bad_parent=True)

                def _fetch_ekd ():
                    # tmpdir = Path(tempfile.mkdtemp(prefix="ekd_"))
                    # out = tmpdir / "data.nc"
                    # print("Save to tmp file:", out)
                    return src_ekd.to_xarray(**(self.to_xarray_args or {}))

                # print(src_ekd)
                # print(self.to_xarray_args)
                ds_chunk = retry_fetch_after_hdf_err(_fetch_ekd, error_re=r"NetCDF:.*HDF error", base_sleep=2, delete_bad_file=False)
                # print(ds_chunk)

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
                if self.request_type in ("daily", "hourly"):
                    if self.provider == "ecmwf-open-data":
                        time_freq = generate_hours(self.data_selection.period.freq, 'int')
                    else:
                        time_freq = generate_hours(self.data_selection.period.freq)
                    request_time_args = dict(
                        date=f"{y1:%Y-%m-%d}/{y2:%Y-%m-%d}",
                        time=time_freq,
                    )
                    request_time_args_list.append(request_time_args)
                elif self.request_type == "monthly":
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
            ds_all = xr.concat(datasets, dim=xarray_concat_dim, **self.xarray_concat_extra_args)

        else:
            request_time_args_list = []
            if self.request_type in ("daily", "hourly"):
                if self.provider == "ecmwf-open-data":
                    time_freq = generate_hours(self.data_selection.period.freq, 'int')
                else:
                    time_freq = generate_hours(self.data_selection.period.freq)
                request_time_args = dict(
                    date=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                    time=time_freq,
                )
                request_time_args_list.append(request_time_args)
            elif self.request_type == "monthly":
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
            ds_all = xr.concat(datasets, dim=xarray_concat_dim, **self.xarray_concat_extra_args)

        # Drop unused variables
        ds_all = ds_all.drop_vars([v for v in ds_all.data_vars if v not in self.var_name_list])

        # Drop missing samples
        xarray_concat_dim = _guess_dim_name(ds_all, "time", ["valid_time", "time_counter"]) if not self.xarray_concat_dim else self.xarray_concat_dim
        if self.elements.missed:
            ds_all = ds_all.drop_sel({xarray_concat_dim: list(self.elements.missed)}, errors='ignore')

        # Add missed info to dataset
        missed_np = np.array(sorted(self.elements.missed), dtype="datetime64[ns]")
        # print(missed_np)
        ds_all = ds_all.assign_coords(missed_time=("missed_time", missed_np))
        ds_all["missed_time"].encoding.update({
            "units": "nanoseconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        })

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
