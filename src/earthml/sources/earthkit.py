from typing import Literal, Any, Callable
from dataclasses import dataclass, field

import os, time, random
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path
import hashlib, json

from contextlib import contextmanager

from datetime import timedelta, datetime, timezone
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY

import numpy as np
import pandas as pd
import xarray as xr
import earthkit.data as ekd

from .dataclasses import DataSource, RegridConfig
from .utils import retry_fetch_after_hdf_err, generate_hours
from .base import BaseSource
from ..base import leadtime_to_timedelta
from ..logging import get_logger


logger = get_logger(__name__)


@dataclass
class EarthkitSourceConfig:
    leadtime                    : relativedelta
    provider                    : str
    dataset                     : str
    regrid_config               : RegridConfig
    split_request               : bool = False
    split_month                 : int = 12
    split_month_jump            : list[str] | None = None
    select_area_after_request   : bool = False
    request_type                : Literal["hourly", "daily", "monthly"] = "hourly"
    request_extra_args          : dict[str, Any] = field(default_factory=dict)
    to_xarray_args              : dict[str, Any] = field(default_factory=dict)
    xarray_concat_dim           : str | None = None
    xarray_concat_extra_args    : dict[str, Any] = field(default_factory=dict)
    convert_unit                : dict[str, tuple[Callable, str]] = field(default_factory=dict)
    earthkit_cache_dir          : Path = Path("/tmp/earthkit-cache/")


class EarthkitSource(BaseSource):
    """
    Collect data using ECMWF new earthkit library.
    """
    def __init__(
        self,
        datasource  : DataSource,
        config      : EarthkitSourceConfig
    ):
        super().__init__(datasource)
        self.config = config

        self.ekd_version = ekd.__version__
        # Shared, persistent cache directory (reused across restarts)
        ekd_base_cache = Path(config.earthkit_cache_dir)
        ekd_base_cache.mkdir(parents=True, exist_ok=True)

        ekd.config.set("cache-policy", "user")
        ekd.config.set("user-cache-directory", str(ekd_base_cache))

        # Locks must be shared across jobs/runs if cache is shared
        self._lock_dir = ekd_base_cache / ".locks"
        self._lock_dir.mkdir(parents=True, exist_ok=True)

        self.elements.samples = self.date_range

        self.split_month_jump = config.split_month_jump if config.split_month_jump else []

        self.select_area_after_request = self.config.select_area_after_request
        self.convert_unit = self.config.convert_unit if self.config.convert_unit else None

        self.regrid_resolution = self.config.regrid_config.regrid_resolution
        self.regrid_vars = config.regrid_config.regrid_vars if config.regrid_config.regrid_vars is not None else self.var_name_list

        self._create_leadtime_dict()
        self._populate_missed()


    def _populate_missed(self):
        """Populate missed if some months are skipped for seasonal requests"""
        # TODO we only support monthly seasonal datasets
        if self.config.request_type == "monthly":
            # Use effective start/end
            start = self.date_range[0] - self.config.leadtime
            end = self.date_range[-1] - self.config.leadtime
            skip_months = set(self.split_month_jump)

            missed = [
                dt + self.config.leadtime for dt in rrule(MONTHLY, dtstart=start, until=end)
                if f"{dt.month:02d}" in skip_months
            ]
            # Snap all timesteps to month-start (00:00 of the 1st)
            t = pd.to_datetime(missed)
            self.elements.missed = set(t.to_period("M").to_timestamp(how="start"))  # month begin, midnight


    def _create_leadtime_dict(self):
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
                td = leadtime_to_timedelta(lt)
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


    @contextmanager
    def _file_lock(self, lock_path: Path, timeout_s: int = 600, poll_s: float = 0.2, stale_s: int = 6 * 3600):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()

        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(f"pid={os.getpid()}\ncreated={time.time()}\n")
                break
            except FileExistsError:
                # stale lock cleanup
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > stale_s:
                        lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue

                if time.time() - start > timeout_s:
                    raise TimeoutError(f"Timeout waiting for lock: {lock_path}")
                time.sleep(poll_s + random.random() * 0.1)

        try:
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


    def _fetch_chunks(self, request_time_args_list, start, end, request_other_args):
        """
        Helper to fetch chunked datasets using ekd
        """
        def _format_request_chunk_log(req_time_arg, start, end, idx, total):
            months_req = req_time_arg.get("month")
            years_req = req_time_arg.get("year")
            date_req = req_time_arg.get("date")
            time_req = req_time_arg.get("time")

            parts = [f" → Fetching chunk {idx}/{total}:"]

            if date_req is not None:
                parts.append(f"date={date_req}")
            else:
                parts.append(f"window={start:%Y-%m-%d}..{end:%Y-%m-%d}")

            if years_req is not None:
                parts.append(f"year={years_req}")
            if months_req is not None:
                parts.append(f"month={months_req}")
            if time_req is not None:
                parts.append(f"time={time_req}")

            return ", ".join(parts)

        ds_chunks = []
        for i, req_time_arg in enumerate(request_time_args_list, start=1):
            logger.info("%s", _format_request_chunk_log(req_time_arg, start, end, i, len(request_time_args_list)))

            request_d = dict(
                **request_other_args,
                **req_time_arg,
            )
            # print(request_d)

            if self.config.dataset:
                src_ekd_params = {"name": self.config.provider, "dataset": self.config.dataset} | request_d
                # src_ekd = ekd.from_source(self.config.provider, self.config.dataset, **request_d)
            else:
                src_ekd_params = {"name": self.config.provider} | request_d

            def _freeze(obj):
                """Convert obj into a JSON-serializable, order-stable structure."""
                if isinstance(obj, dict):
                    return {str(k): _freeze(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
                if isinstance(obj, (list, tuple, set)):
                    return [_freeze(v) for v in obj]
                if isinstance(obj, np.ndarray):
                    return [_freeze(v) for v in obj.tolist()]
                if hasattr(obj, "tolist"):  # xarray/pandas scalars
                    try:
                        return _freeze(obj.tolist())
                    except Exception:
                        pass
                # common scalar types
                if isinstance(obj, (str, int, float, bool)) or obj is None:
                    return obj
                # datetime-like / timedelta-like
                try:
                    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                        return str(obj)
                except Exception:
                    pass
                # fallback: string repr (last resort, but stable enough for lock key)
                return repr(obj)

            def _lock_key_from_params(params: dict) -> str:
                frozen = _freeze(params)
                payload = json.dumps(frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                return hashlib.sha256(payload.encode("utf-8")).hexdigest()

            lock_key = _lock_key_from_params(src_ekd_params)
            lock_path = self._lock_dir / f"{lock_key}.lock"

            def _fetch_ekd_src():
                with self._file_lock(lock_path):
                    return ekd.from_source(**src_ekd_params)

            src_ekd = retry_fetch_after_hdf_err(
                _fetch_ekd_src,
                error_re=r"NetCDF:.*HDF error",
                base_sleep=5,
                tries=5,
                delete_bad_file=False,
                delete_bad_parent=False,
            )

            def _fetch_ekd():
                with self._file_lock(lock_path): # TODO safe to reuse the same lock?
                    # Open and load dataset
                    ds_chunk = src_ekd.to_xarray(**(self.config.to_xarray_args or {}))
                    ds_chunk = ds_chunk.load()

                    # Validate quickly: expected variables present
                    missing = [v for v in self.var_name_list if v not in ds_chunk.data_vars]
                    if missing:
                        raise ValueError(f"Missing variables in chunk: {missing}. Got: {list(ds_chunk.data_vars)}")

                    # Touch a tiny slice to force a real read
                    probe_var = self.var_name_list[0]
                    _ = ds_chunk[probe_var].isel(
                        {d: 0 for d in ds_chunk[probe_var].dims if ds_chunk.sizes.get(d, 1) > 0}
                    ).load()

                    return ds_chunk

            # print(src_ekd)
            # print(self.config.to_xarray_args)
            ds_chunk = retry_fetch_after_hdf_err(
                _fetch_ekd,
                error_re=r"NetCDF:.*HDF error",
                base_sleep=2,
                tries=5,
                delete_bad_file=True,
                delete_bad_parent=True,
            )

            # Validate quickly: expected variables present
            missing = [v for v in self.var_name_list if v not in ds_chunk.data_vars]
            if missing:
                raise ValueError(f"Missing variables in chunk: {missing}. Got: {list(ds_chunk.data_vars)}")

            # Touch a tiny slice to force a real read
            probe_var = self.var_name_list[0]
            _ = ds_chunk[probe_var].isel({d: 0 for d in ds_chunk[probe_var].dims if ds_chunk.sizes.get(d, 1) > 0}).load()

            # print(ds_chunk)

            realization_dim = ds_chunk.earthml.guessed_dims.realization
            leadtime_dim = ds_chunk.earthml.guessed_dims.leadtime # actual leadtime in chunk

            logger.info("   Chunk size: %s", ds_chunk.sizes)
            # print(f"   Chunk coords: {ds_chunk.coords}")
            # print(f"   Leadtime: {self.leadtime_d}")

            # TODO Control thoroughly this flow
            if self.leadtime_d:
                # Get requested leadtime and coord name
                target_lt_d = next(iter(self.leadtime_d.values()))
                leadtime_name = next(iter(self.leadtime_d.keys())) # leadtime dim name we want

                if target_lt_d is not None and leadtime_dim in ds_chunk.coords:
                    # Normalize requested target to the coordinate dtype
                    if not isinstance(target_lt_d, pd.Timedelta):
                        target_lt_pd = pd.to_timedelta(target_lt_d)
                    else:
                        target_lt_pd = target_lt_d

                    coord_dtype = ds_chunk[leadtime_dim].dtype

                    # For timedelta64 coords: convert via to_timedelta64 then cast to coord dtype
                    # For numeric coords this still works if target_lt_pd is numeric-like
                    try:
                        target_np = np.array(target_lt_pd.to_timedelta64(), dtype=coord_dtype)
                    except Exception:
                        # Fallback generic cast (useful if coords are ints/floats)
                        target_np = np.asarray(target_lt_pd, dtype=coord_dtype)

                    # Find nearest available leadtime in this chunk
                    lt_values = ds_chunk[leadtime_dim].values

                    # print(f"    Leadtime values: {lt_values}")
                    unique_lt = np.unique(lt_values)

                    # Target_np should be same dtype as unique_lt (timedelta or numeric)
                    idx0 = np.argmin(np.abs(unique_lt - target_np))
                    center_lt = unique_lt[idx0]

                    # Window around nearest center
                    half_window = np.timedelta64(3, "D") # 3-day window is a good guess # TODO maybe improve? make it explicit?
                    low, high = center_lt - half_window, center_lt + half_window

                    mask_lt = (lt_values >= low) & (lt_values <= high)
                    idxs = np.where(mask_lt)[0]

                    logger.info(
                        "   Nearest leadtime center: %s, window: [%s, %s], inferred realizations %s",
                        pd.to_timedelta(center_lt) if np.issubdtype(unique_lt.dtype, np.timedelta64) else center_lt,
                        pd.to_timedelta(low) if np.issubdtype(unique_lt.dtype, np.timedelta64) else low,
                        pd.to_timedelta(high) if np.issubdtype(unique_lt.dtype, np.timedelta64) else high,
                        idxs.size,
                    )

                    if idxs.size == 0:
                        raise ValueError("No leadtimes found in the computed window")

                    ds_chunk = ds_chunk.isel({leadtime_dim: idxs})
                    # print(f"   Chunk size after selection: {ds_chunk.sizes}")

                    # TODO realization guessing is brittle, but I have no alternative for now
                    if leadtime_dim in ds_chunk.dims and realization_dim is None:
                        # In this case leadtime is the time dimension, swap and sort
                        ds_chunk = ds_chunk.swap_dims({leadtime_dim: "time"})
                        if leadtime_dim in ds_chunk.data_vars:
                            ds_chunk = ds_chunk.drop_vars(leadtime_dim)
                        ds_chunk = ds_chunk.sortby("time")
                        # print(f"   Chunk coords after swap: {ds_chunk.coords}")

                        # Infer member index from repeats within each time
                        # This assumes that for each valid time we have the same number of stacked members.
                        t = ds_chunk["time"].values
                        unique_t, counts = np.unique(t, return_counts=True)

                        if counts.min() != counts.max():
                            raise ValueError(f"Unequal members per time: min={counts.min()} max={counts.max()} counts={dict(zip(unique_t, counts))}")

                        n_realizations = int(counts[0])
                        realization_index = np.concatenate([np.arange(n_realizations, dtype=np.int32) for _ in range(len(unique_t))])
                        # print(f"    Inferred realization number: {n_realizations}, index: {realization_index}")

                        # Add member coordinate and unstack to get dimensions (time, realization)
                        ds_chunk = ds_chunk.assign_coords(realization=("time", realization_index))
                        # print(f"   Realization coord: {ds_chunk["realization"].values}")

                        # Avoid name collision
                        ds_chunk = ds_chunk.rename_vars({"time": "valid_time"})

                        # MultiIndex
                        ds_chunk = ds_chunk.set_index(time=["valid_time", "realization"]).unstack("time")

                        # Rename time dim
                        ds_chunk = ds_chunk.rename({"valid_time": "time"})

                        # Drop variable only if it exists as a data variable (avoid dropping the coord unintentionally)
                        if leadtime_dim in ds_chunk.data_vars:
                            ds_chunk = ds_chunk.drop_vars(leadtime_dim)

                        # Reassign the coordinate to the requested target (ensures uniform coord across chunks)
                        ds_chunk = ds_chunk.assign_coords({leadtime_name: (leadtime_name, np.array([target_np], dtype=coord_dtype))})

                    else:
                        # if ds_chunk.sizes.get(leadtime_name, 0) > 1:
                        # Move leadtime first to make bfill deterministic across the step axis
                        ds_chunk = ds_chunk.transpose(leadtime_dim, ...)

                        # Since at most one is non-NaN, bfill then take leadtime=0 gives the only valid value
                        ds_chunk = ds_chunk.bfill(leadtime_dim).isel({leadtime_dim: 0})

                        # Restore a length-1 leadtime dimension with the requested conceptual value
                        ds_chunk = ds_chunk.expand_dims({leadtime_name: [target_np]})

                    logger.info("   Size after all processing: %s", ds_chunk.sizes)

            xarray_concat_dim = ds_chunk.earthml.guessed_dims.time if not self.config.xarray_concat_dim else self.config.xarray_concat_dim
            # print(xarray_concat_dim)
            ds_chunks.append(ds_chunk)
        return xarray_concat_dim, ds_chunks

    def _get_data(self) -> xr.Dataset:
        n_missed = len(self.elements.missed) # at this stage is always zero
        samples = [s for s in self.elements.samples if s not in self.elements.missed] # TODO refactor to BaseSource?
        logger.info("Samples: %s, missed: %s", len(samples), n_missed)

        var_longname_list = [v.longname for v in self.data_selection.variable] if isinstance(self.data_selection.variable, list) else [self.data_selection.variable.longname]

        # print(f"Earthkit period shifted: {self.data_selection.period.shifted}")

        # print(f"Lead time: {self.config.leadtime}")
        lead_is_zero = (
            self.config.leadtime.years == 0
            and self.config.leadtime.months == 0
            and self.config.leadtime.days == 0
            and self.config.leadtime.hours == 0
            and self.config.leadtime.minutes == 0
            and self.config.leadtime.seconds == 0
        )
        # lead_is_zero = self.config.leadtime == relativedelta() # possibile stricter alternative

        # print(f"Leadtime is zero: {lead_is_zero}")
        # print(f"Data sel start: {self.date_range[0]}, end: {self.date_range[-1]}")
        # Use effective start and end dates
        start = self.date_range[0] - self.config.leadtime #+ relativedelta(**self.data_selection.period.shifted) if self.data_selection.period.shifted is not None else self.data_selection.period.start
        end = self.date_range[-1] - self.config.leadtime #+ relativedelta(**self.data_selection.period.shifted) if self.data_selection.period.shifted is not None else self.data_selection.period.end

        # TODO wrong if months requested are 12 1, it splits wrongly
        dates = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        all_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        months_splitted = [
            [m for m in all_months[i:i+self.config.split_month] if m not in self.split_month_jump]
            for i in range(0, len(all_months), self.config.split_month)
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
        if self.config.select_area_after_request:
            request_args = dict(variable=var_longname_list)
        else:
            request_args = dict(variable=var_longname_list, area=area)

        years = xr.date_range(start=start, end=end, freq="YS") # from the first year after the start
        if len(years) == 0:
            raise ValueError("Years calculated from date range is empty")

        logger.info(
            "Requesting %s (%s, %s) in region %s from %s:%s (ekd_ver=%s)",
            var_longname_list,
            dates,
            self.data_selection.period.freq,
            area,
            self.config.provider,
            self.config.dataset,
            self.ekd_version,
        )
        logger.info("Check request status: https://cds.climate.copernicus.eu/requests?tab=all")

        if (
            self.config.split_request and (
                end - start > pd.to_timedelta('365 days') or
                end.year == start.year+1
            )
        ):
            # print(f"Years before change: {years}")
            # Split into yearly chunks
            year_first, year_last = years[0], years[-1]
            if year_first > start:
                years = xr.date_range(start, start, periods=1).append(years)
            if year_last <= end:
                if self.config.request_type in ("daily", "hourly"):
                    years = years.append(
                        xr.date_range(end + timedelta(days=1), end + timedelta(days=1), periods=1)
                    )
                elif self.config.request_type == "monthly":
                    years = years.append(
                        xr.date_range(end + relativedelta(months=1), end + relativedelta(months=1), periods=1)
                    )
            logger.info("Requested split-by-year ranges: %s", years)

            datasets = []
            xarray_concat_dim = None
            for y1, y2 in zip(years[:-1], years[1:]):
                # print(f"y1={y1}, y2={y2}")
                request_time_args_list = []
                if self.config.request_type in ("daily", "hourly"):
                    y2 = y2 - timedelta(days=1) # inclusive end
                    if self.config.provider == "ecmwf-open-data":
                        time_freq = generate_hours(self.data_selection.period.freq, 'int')
                    else:
                        time_freq = generate_hours(self.data_selection.period.freq)
                    request_time_args = dict(
                        date=f"{y1:%Y-%m-%d}/{y2:%Y-%m-%d}",
                        time=time_freq,
                    )
                    request_time_args_list.append(request_time_args)

                elif self.config.request_type == "monthly":
                    y22 = y2 - relativedelta(months=1) # inclusive end
                    y22 = y22 - relativedelta(years=1) if y22.strftime("%Y") != y1.strftime("%Y") else y22
                    # print(f"y22={y22}")
                    split_req_months = [m.strftime("%m") for m in xr.date_range(start=y1, end=y22, freq="MS")]
                    # print(f"split_req_months: {split_req_months}")
                    for m in months_splitted:
                        # print(f"   m:{m}")
                        request_time_args = dict(year=xr.date_range(start=datetime(y1.year, 1, 1), end=datetime(y22.year, 1, 1), freq="YS").strftime("%Y").tolist())
                        if "month" not in self.config.request_extra_args:
                            if isinstance(m, list):
                                month_req = [x for x in m if x in set(split_req_months)]
                            else:
                                if m in split_req_months:
                                    month_req = m
                                else:
                                    continue
                            request_time_args['month'] = month_req
                        request_time_args_list.append(request_time_args)

                else:
                    raise ValueError(f"Unsupported earthkit request type {self.config.request_type}")

                # print(f"   request_time_args_list: {request_time_args_list}")
                if request_time_args_list:
                    xarray_concat_dim, ds_chunks = self._fetch_chunks(request_time_args_list, y1, y2, request_args | self.config.request_extra_args)
                    datasets.extend(ds_chunks)

            # Guard for empy datasets and no concat dim
            if not datasets or xarray_concat_dim is None:
                raise ValueError(
                    f"No datasets fetched for {self.source_name}. "
                    f"request_type={self.config.request_type}, start={start}, end={end}"
                )

            # Combine all datasets
            # print(f"Combine split-request datasets of length {len(datasets)}")
            ds_all = xr.concat(datasets, dim=xarray_concat_dim, **self.config.xarray_concat_extra_args)
            # print(f"Combined datasets: {len(ds_all[xarray_concat_dim].values)}")

        else:
            request_time_args_list = []
            if self.config.request_type in ("daily", "hourly"):
                if self.config.provider == "ecmwf-open-data":
                    time_freq = generate_hours(self.data_selection.period.freq, 'int')
                else:
                    time_freq = generate_hours(self.data_selection.period.freq)
                request_time_args = dict(
                    date=f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                    time=time_freq,
                )
                request_time_args_list.append(request_time_args)
            elif self.config.request_type == "monthly":
                # print(f"start: {start}, end: {end}")
                split_req_months = [m.strftime("%m") for m in xr.date_range(start=start, end=end, freq="MS")]
                # print(split_req_months)
                for m in months_splitted:
                    request_time_args = dict(year=xr.date_range(start=datetime(start.year, 1, 1), end=datetime(end.year, 1, 1), freq="YS").strftime("%Y").tolist())
                    if "month" not in self.config.request_extra_args:
                        if isinstance(m, list):
                            month_req = [x for x in m if x in set(split_req_months)]
                        else:
                            if m in split_req_months:
                                month_req = m
                            else:
                                continue
                        request_time_args['month'] = month_req
                    # print(request_time_args)
                    request_time_args_list.append(request_time_args)
            else:
                raise ValueError(f"Unsupported earthkit request type {self.config.request_type}")
            if request_time_args_list:
                xarray_concat_dim, datasets = self._fetch_chunks(request_time_args_list, start, end, request_args | self.config.request_extra_args)

            # Guard for empy datasets and no concat dim
            if not datasets or xarray_concat_dim is None:
                raise ValueError(
                    f"No datasets fetched for {self.source_name}. "
                    f"request_type={self.config.request_type}, start={start}, end={end}"
                )

            # Combine all datasets
            ds_all = xr.concat(datasets, dim=xarray_concat_dim, **self.config.xarray_concat_extra_args)

        # Drop unused variables
        ds_all = ds_all.drop_vars([v for v in ds_all.data_vars if v not in self.var_name_list])

        # Ensure concat dim is datetime64[ns]
        ds_all = xr.decode_cf(ds_all)
        ds_all = ds_all.assign_coords({xarray_concat_dim: ds_all[xarray_concat_dim].astype("datetime64[ns]")})
        # print(f"Time values before reassignment: {ds_all[xarray_concat_dim].values}")

        if self.config.request_type == "monthly":
            # Snap all timesteps to month-start (00:00 of the 1st)
            t = pd.to_datetime(ds_all[xarray_concat_dim].values)
            t_month_start = t.to_period("M").to_timestamp(how="start")  # month begin, midnight
            # print(t_month_start.values)
            ds_all = ds_all.assign_coords({xarray_concat_dim: (xarray_concat_dim, t_month_start.values)})
            # Optional: unique after snapping (avoid duplicates)
            # ds_all = ds_all.groupby(xarray_concat_dim).first()

        # print(f"Datasets before dropping missing samples: {len(ds_all[xarray_concat_dim].values)}")
        # print(f"Time values after reassignment: {ds_all[xarray_concat_dim].values}")

        n_missed = len(self.elements.missed) # recompute after fetching
        n_actual_samples = len(ds_all[xarray_concat_dim].values)
        n_all_samples = len(self.date_range)
        assert n_missed + n_actual_samples == n_all_samples, f"number of samples obtained + missed is different from number of all samples ({n_actual_samples}+{n_missed} =/= {n_all_samples})"
        # print(f"Missed: {self.elements.missed}")
        # Drop missing samples
        xarray_concat_dim = ds_all.earthml.guessed_dims.time if not self.config.xarray_concat_dim else self.config.xarray_concat_dim
        if self.elements.missed:
            ds_all = ds_all.drop_sel({xarray_concat_dim: list(self.elements.missed)}, errors='ignore')

        # print(f"Datasets after dropping missing samples: {len(ds_all[xarray_concat_dim].values)}")

        # Shift time index by lead time to get appropriate time corresponding for both forecast and analysis
        # time_index = pd.DatetimeIndex(ds_all[xarray_concat_dim].values)
        # if self.config.request_type in ("daily", "hourly"):
        #     shift_delta = relativedelta(days=1)
        # elif self.config.request_type == "monthly":
        #     shift_delta = relativedelta(months=1)
        # shifted = time_index.map(lambda t: t - self.config.leadtime + shift_delta) # +1 day/month to compensate for inclusive end in requests (e.g. 2020-01-31 is included in Jan request, but after shifting it becomes 2020-01-30 which is not included in Feb request)

        # actual = pd.to_datetime(ds_all[xarray_concat_dim].values).to_period("M").to_timestamp()
        # expected = pd.to_datetime(self.date_range).to_period("M").to_timestamp()

        # if not actual.equals(expected):
        #     raise ValueError(
        #         f"Forecast valid times do not match expected target dates.\n"
        #         f"Actual:   {list(actual)}\n"
        #         f"Expected: {list(expected)}"
        #     )

        # Canonical convention for earthkit sources:
        # - `time` is always the model-aligned target/valid date used downstream.
        # - raw provider timestamps are preserved separately as `source_time`.
        source_time = pd.DatetimeIndex(pd.to_datetime(ds_all[xarray_concat_dim].values))
        canonical_time = pd.DatetimeIndex(
            [d for d in self.date_range if d not in self.elements.missed]
        )

        if len(source_time) != len(canonical_time):
            raise ValueError(
                "Fetched sample count does not match canonical non-missed date_range.\n"
                f"Fetched count: {len(source_time)}\n"
                f"Canonical count: {len(canonical_time)}\n"
                f"Fetched head/tail: {source_time[:3].tolist()} ... {source_time[-3:].tolist()}\n"
                f"Canonical head/tail: {canonical_time[:3].tolist()} ... {canonical_time[-3:].tolist()}"
            )

        if not source_time.is_monotonic_increasing:
            raise ValueError(
                "Fetched source_time is not monotonic increasing.\n"
                f"Head/tail: {source_time[:3].tolist()} ... {source_time[-3:].tolist()}"
            )

        if not source_time.is_unique:
            duplicates = source_time[source_time.duplicated()].unique().tolist()
            raise ValueError(
                "Fetched source_time contains duplicates after leadtime selection.\n"
                f"Duplicate values (up to 10): {duplicates[:10]}"
            )

        ds_all = ds_all.assign_coords(source_time=(xarray_concat_dim, source_time.values))
        ds_all["source_time"].attrs.update({
            "long_name": "raw provider time coordinate before canonical remapping",
            "note": (
                "For forecast products this may represent initialization time or provider-specific "
                "valid-time semantics depending on dataset/engine. The canonical training axis is `time`."
            ),
        })

        # Assign canonical target/valid dates used everywhere else in the pipeline.
        ds_all = ds_all.assign_coords({xarray_concat_dim: canonical_time.values})

        logger.info(
            "Time convention mapping: source_time head/tail=%s ... %s | canonical_time head/tail=%s ... %s",
            source_time[:3].tolist(),
            source_time[-3:].tolist(),
            canonical_time[:3].tolist(),
            canonical_time[-3:].tolist(),
        )
        if len(source_time) > 0 and len(canonical_time) > 0:
            logger.info(
                "Leadtime audit: requested_leadtime=%s, first source->canonical=%s -> %s",
                self.config.leadtime,
                source_time[0],
                canonical_time[0],
            )
        # Reindex, set NaNs instead of missed values
        # full_index = pd.DatetimeIndex(self.date_range)
        # ds_all = ds_all.reindex({xarray_concat_dim: full_index})

        logger.info(
            "First and last time values in obtained dataset: %s, %s",
            pd.to_datetime(ds_all[xarray_concat_dim].values[0]),
            pd.to_datetime(ds_all[xarray_concat_dim].values[-1]),
        )

        return ds_all
