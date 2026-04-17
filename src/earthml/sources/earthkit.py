from typing import Literal, Any, TypeAlias
from collections.abc import Mapping, Sequence, Callable
from dataclasses import dataclass, field

import os, time, random
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path
import hashlib, json

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

from contextlib import contextmanager

from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule, MONTHLY

import numpy as np
import pandas as pd
import xarray as xr
import earthkit.data as ekd

from .dataclasses import DataSource, RegridConfig
from .utils import retry_fetch_after_hdf_err, generate_hours
from .base import BaseSource
from ..base import Leadtime, Variable
from ..logging import get_logger


logger = get_logger(__name__)
SOURCE_CTX = "[source earthkit]"

LeadtimeEarthkit: TypeAlias = dict[Variable, dict[Literal["request", "dataset"], Leadtime]]

@dataclass
class EarthkitSourceConfig:
    leadtime                    : Leadtime # request leadtime, we only support one for now
    provider                    : str
    dataset                     : str
    regrid_config               : RegridConfig
    split_request               : bool = False
    split_month                 : int = 12
    split_month_jump            : list[str] | None = None
    select_area_after_request   : bool = False
    request_type                : Literal["hourly", "daily", "monthly"] = "hourly"
    request_extra_args          : dict[str, Any] = field(default_factory=dict)
    request_delta_hack          : dict[str, relativedelta] | relativedelta | None = None
    to_xarray_args              : dict[str, Any] = field(default_factory=dict)
    xarray_concat_dim           : str | None = None
    xarray_concat_extra_args    : dict[str, Any] = field(default_factory=dict)
    convert_unit                : dict[str, tuple[Callable, str]] = field(default_factory=dict)
    snap_time_index             : Literal["previous", "same", "next", "nearest"] = "nearest"
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

        self._ekd_setup()

        self.elements.samples = self.date_range

        self.split_month_jump = config.split_month_jump if config.split_month_jump else []

        self.select_area_after_request = self.config.select_area_after_request
        self.convert_unit = self.config.convert_unit if self.config.convert_unit else None

        self.regrid_resolution = self.config.regrid_config.regrid_resolution
        self.regrid_vars = config.regrid_config.regrid_vars if config.regrid_config.regrid_vars is not None else self.var_name_list

        self._init_leadtime_variables_periods() # build self.leadtime_d dict, "request" from config, "dataset" from data selection
        self._populate_missed()


    def _ekd_setup(self):
        """Earthkit-data setup"""
        self.ekd_version = ekd.__version__

        # Shared, persistent cache directory (reused across restarts)
        self.ekd_base_cache = Path(self.config.earthkit_cache_dir)
        self.ekd_base_cache.mkdir(parents=True, exist_ok=True)
        ekd.config.set("cache-policy", "user")
        ekd.config.set("user-cache-directory", str(self.ekd_base_cache))

        # Locks must be shared across jobs/runs if cache is shared
        self._lock_dir = self.ekd_base_cache / ".locks"
        self._lock_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_request_chunk_label(
        req_args: Mapping[str, Any],
        start: pd.Timedelta | datetime,
        end: pd.Timedelta | datetime,
        idx: int,
        total: int,
    ) -> str:
        months_req = req_args.get("month")
        years_req = req_args.get("year")
        date_req = req_args.get("date")
        parts = [f"{idx}/{total}"]

        if date_req is not None:
            parts.append(f"date {date_req}")
        else:
            parts.append(f"{start:%Y-%m}..{end:%Y-%m}")

        if years_req is not None:
            if isinstance(years_req, Sequence) and not isinstance(years_req, str):
                years_req = ",".join(str(y) for y in years_req)
            parts.append(f"y={years_req}")
        if months_req is not None:
            if isinstance(months_req, Sequence) and not isinstance(months_req, str):
                month_values = [str(m) for m in months_req]
                months_repr = month_values[0] if len(month_values) == 1 else f"{month_values[0]}-{month_values[-1]}"
            else:
                months_repr = str(months_req)
            parts.append(f"m={months_repr}")

        return ", ".join(parts)

    @staticmethod
    def _format_progress_stage(stage: str | None) -> str | None:
        if not stage:
            return None

        if stage.startswith("earthkit-data"):
            return "download"

        normalized = {
            "prepare request": "prepare",
            "submit request": "submit",
            "load xarray": "decode",
            "chunk complete": "finalize",
        }
        return normalized.get(stage, stage)

    @staticmethod
    def _set_progress(
        progress: Progress | None,
        task_id: int | None,
        *,
        total_chunks: int,
        chunk_position: float | None = None,
        stage: str | None = None,
        chunk_label: str | None = None,
    ) -> None:
        if progress is None or task_id is None:
            return

        update_kwargs: dict[str, Any] = {}
        if chunk_position is not None:
            update_kwargs["completed"] = max(0.0, min(float(chunk_position), float(total_chunks)))

        if chunk_label is not None:
            description = f"{SOURCE_CTX} {chunk_label}"
            if stage:
                formatted_stage = EarthkitSource._format_progress_stage(stage)
                if formatted_stage:
                    description = f"{description} [{formatted_stage}]"
            update_kwargs["description"] = description

        if update_kwargs:
            progress.update(task_id, **update_kwargs)

    def _progress_task_description(
        self,
        variable_name: str,
        date_range_text: str,
    ) -> str:
        return f"{SOURCE_CTX} {variable_name} {date_range_text}"

    @contextmanager
    def _earthkit_progress_redirect(
        self,
        progress: Progress | None,
        task_id: int | None,
        *,
        chunk_index: int,
        total_chunks: int,
        chunk_label: str,
    ):
        if progress is None or task_id is None:
            yield
            return

        import earthkit.data.sources.url as ekd_url
        import earthkit.data.utils.progbar as ekd_progbar
        import cdsapi.api as cdsapi_api
        import logging

        orig_progress_bar = ekd_progbar.progress_bar
        orig_tqdm = ekd_progbar.tqdm
        orig_url_progress_bar = ekd_url.progress_bar
        orig_cdsapi_tqdm = cdsapi_api.tqdm
        cdsapi_logger = logging.getLogger("cdsapi")
        orig_cdsapi_disabled = cdsapi_logger.disabled
        orig_cdsapi_level = cdsapi_logger.level
        base_completed = max(chunk_index - 1, 0)

        class _RichEarthkitProgress:
            def __init__(self, *, total=None, iterable=None, initial=0, desc=None, **_kwargs):
                self.total = total
                self.iterable = iterable
                self.current = initial
                self.desc = desc
                self._refresh()

            def _stage(self) -> str:
                if self.desc:
                    return f"earthkit-data: {self.desc}"
                return "earthkit-data"

            def _refresh(self) -> None:
                if self.total:
                    fraction = min(max(self.current / self.total, 0.0), 1.0)
                    EarthkitSource._set_progress(
                        progress,
                        task_id,
                        total_chunks=total_chunks,
                        chunk_position=base_completed + fraction,
                        stage=self._stage(),
                        chunk_label=chunk_label,
                    )
                else:
                    EarthkitSource._set_progress(
                        progress,
                        task_id,
                        total_chunks=total_chunks,
                        stage=self._stage(),
                        chunk_label=chunk_label,
                    )

            def update(self, n=1):
                self.current += n
                self._refresh()

            def refresh(self):
                self._refresh()

            def close(self):
                return None

            def set_description(self, desc=None, refresh=True):
                self.desc = desc
                if refresh:
                    self._refresh()

            def __iter__(self):
                if self.iterable is None:
                    return iter(())
                for item in self.iterable:
                    yield item
                    self.update(1)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()
                return False

        def _progress_factory(*, total=None, iterable=None, initial=0, desc=None, **kwargs):
            return _RichEarthkitProgress(
                total=total,
                iterable=iterable,
                initial=initial,
                desc=desc,
                **kwargs,
            )

        ekd_progbar.progress_bar = _progress_factory
        ekd_progbar.tqdm = _RichEarthkitProgress
        ekd_url.progress_bar = _progress_factory
        cdsapi_api.tqdm = _RichEarthkitProgress
        cdsapi_logger.disabled = True
        cdsapi_logger.setLevel(logging.WARNING)
        try:
            yield
        finally:
            ekd_progbar.progress_bar = orig_progress_bar
            ekd_progbar.tqdm = orig_tqdm
            ekd_url.progress_bar = orig_url_progress_bar
            cdsapi_api.tqdm = orig_cdsapi_tqdm
            cdsapi_logger.disabled = orig_cdsapi_disabled
            cdsapi_logger.setLevel(orig_cdsapi_level)

    def _init_leadtime_variables_periods(self):
        self.variables = list(self.data_selection.variable) if isinstance(self.data_selection.variable, Sequence) else [self.data_selection.variable]
        leadtime_config = self.config.leadtime
        self.leadtime_d: LeadtimeEarthkit = {}
        for v in self.variables:
            leadtime_var = v.leadtime if v.leadtime is not None else leadtime_config
            self.leadtime_d[v] = {
                "request": leadtime_config,
                "dataset": leadtime_var,
            }
        self.start_d, self.end_d = {}, {}
        for v in self.variables:
            # these are the same, not really per-variable, but in the future maybe
            self.start_d[v] = self.date_range[0] - self.leadtime_d[v]["request"].to_timedelta()
            self.end_d[v] = self.date_range[-1] - self.leadtime_d[v]["request"].to_timedelta()
            logger.debug("%s Periods to be requested for %s: %s - %s", SOURCE_CTX, v.name, self.start_d[v], self.end_d[v])

    # TODO extend support for other requests?
    def _populate_missed(self):
        """Populate missed if some months are skipped for seasonal monthly requests only"""
        if self.config.request_type == "monthly":
            skip_months = set(self.split_month_jump)

            missed = set()
            for v in self.variables:
                missed |= set(
                    dt + self.leadtime_d[v]["request"].to_timedelta() for dt in rrule(MONTHLY, dtstart=self.start_d[v], until=self.end_d[v])
                    if f"{dt.month:02d}" in skip_months
                )
            # Snap all timesteps to month-start (00:00 of the 1st)
            t = pd.to_datetime(sorted(list(missed))).to_period("M").to_timestamp(how="start")
            self.elements.missed = set(t) # month begin, midnight #type: ignore


    @contextmanager
    def _file_lock(
        self,
        lock_path: Path,
        timeout_s: int = 600,
        poll_s: float = 0.2,
        stale_s: int = 6 * 3600
    ):
        """Context manager for an exclusive file lock with timeout and stale-lock cleanup."""
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
                    raise TimeoutError(f"{SOURCE_CTX} Timeout waiting for lock: {lock_path}")
                time.sleep(poll_s + random.random() * 0.1)

        try:
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


    def _fetch_chunks(
        self,
        request_args_list: Sequence[dict],
        start: pd.Timedelta | datetime,
        end: pd.Timedelta | datetime,
        leadtime: Leadtime, # leadtime in the dataset, "dataset" in self.leadtime_d
        request_other_args: dict,
        total_chunks: int | None = None,
        chunk_offset: int = 0,
        progress: Progress | None = None,
        progress_task_id: int | None = None,
    ) -> tuple[str | None, Sequence[xr.Dataset]]:
        """
        Helper to fetch chunked datasets using ekd
        """
        total_chunks = len(request_args_list) if total_chunks is None else total_chunks
        ds_chunks = []
        xarray_concat_dim = None
        for i, req_args in enumerate(request_args_list, start=1):
            chunk_index = i + chunk_offset
            chunk_label = self._format_request_chunk_label(req_args, start, end, chunk_index, total_chunks)
            logger.debug("%s Fetching %s", SOURCE_CTX, chunk_label)
            self._set_progress(
                progress,
                progress_task_id,
                total_chunks=total_chunks,
                chunk_position=chunk_index - 1,
                stage="prepare request",
                chunk_label=chunk_label,
            )

            request_d = dict(
                **request_other_args,
                **req_args,
            )
            logger.debug("   %s Full request to ekd: %s", SOURCE_CTX, request_d)

            if self.config.dataset:
                src_ekd_params: Mapping[str, Any] = {"name": self.config.provider, "dataset": self.config.dataset} | request_d
                # src_ekd = ekd.from_source(self.config.provider, self.config.dataset, **request_d)
            else:
                src_ekd_params: Mapping[str, Any] = {"name": self.config.provider} | request_d

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

            with self._earthkit_progress_redirect(
                progress,
                progress_task_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                chunk_label=chunk_label,
            ):
                self._set_progress(
                    progress,
                    progress_task_id,
                    total_chunks=total_chunks,
                    chunk_position=chunk_index - 1,
                    stage="submit request",
                    chunk_label=chunk_label,
                )
                src_ekd: ekd.Source = retry_fetch_after_hdf_err(
                    _fetch_ekd_src,
                    error_re=r"NetCDF:.*HDF error",
                    base_sleep=5,
                    tries=5,
                    delete_bad_file=False,
                    delete_bad_parent=False,
                    manage_earthkit_cache=True,
                )

            def _fetch_ekd():
                with self._file_lock(lock_path): # reuse of the same lock could affect performances
                    # Open and load dataset
                    ds_chunk: xr.Dataset = src_ekd.to_xarray(**(self.config.to_xarray_args or {}))
                    ds_chunk = ds_chunk.load()

                    # Validate quickly: expected variables present
                    missing = [v for v in self.var_name_list if v not in ds_chunk.data_vars]
                    if missing:
                        raise ValueError(f"{SOURCE_CTX} Missing variables in chunk: {missing}. Got: {list(ds_chunk.data_vars)}")

                    # Touch a tiny slice to force a real read
                    probe_var = self.var_name_list[0]
                    _ = ds_chunk[probe_var].isel(
                        {d: 0 for d in ds_chunk[probe_var].dims if ds_chunk.sizes.get(d, 1) > 0}
                    ).load()

                    return ds_chunk

            # logger.debug(src_ekd)
            # logger.debug(self.config.to_xarray_args)
            with self._earthkit_progress_redirect(
                progress,
                progress_task_id,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                chunk_label=chunk_label,
            ):
                self._set_progress(
                    progress,
                    progress_task_id,
                    total_chunks=total_chunks,
                    chunk_position=chunk_index - 1,
                    stage="load xarray",
                    chunk_label=chunk_label,
                )
                ds_chunk = retry_fetch_after_hdf_err(
                    _fetch_ekd,
                    error_re=r"NetCDF:.*HDF error",
                    base_sleep=2,
                    tries=5,
                    delete_bad_file=True,
                    delete_bad_parent=True,
                    manage_earthkit_cache=True,
                )

            # Validate quickly: expected variables present
            missing = [v for v in self.var_name_list if v not in ds_chunk.data_vars]
            if missing:
                raise ValueError(f"{SOURCE_CTX} Missing variables in chunk: {missing}. Got: {list(ds_chunk.data_vars)}")

            # Touch a tiny slice to force a real read
            probe_var = self.var_name_list[0]
            _ = ds_chunk[probe_var].isel({d: 0 for d in ds_chunk[probe_var].dims if ds_chunk.sizes.get(d, 1) > 0}).load()

            # logger.debug(ds_chunk)

            # Get dims from chunk, could be None
            realization_dim = ds_chunk.earthml.guessed_dims.realization
            leadtime_dim = ds_chunk.earthml.guessed_dims.leadtime # actual leadtime in chunk

            logger.debug("   %s Chunk size: %s", SOURCE_CTX, ds_chunk.sizes)
            logger.debug("   %s Chunk coords: %s", SOURCE_CTX, ds_chunk.coords)
            logger.debug("   %s Requested leadtime: %s", SOURCE_CTX, self.leadtime_d)
            logger.debug("   %s Leadtime dim from chunk: %s", SOURCE_CTX, leadtime_dim)
            logger.debug("   %s Realization dim from chunk: %s", SOURCE_CTX, realization_dim)

            # Leadtime discovery
            leadtime_target_td = leadtime.to_timedelta()
            coord_dtype = (
                ds_chunk[leadtime_dim].dtype
                if leadtime_dim is not None
                else np.dtype("timedelta64[ns]")
            )

            # For timedelta64 coords: convert via to_timedelta64 then cast to coord dtype
            # For numeric coords this still works if leadtime_target_td is numeric-like
            try:
                leadtime_target_np = np.array(leadtime_target_td.to_timedelta64(), dtype=coord_dtype)
            except Exception:
                # Fallback generic cast (useful if coords are ints/floats)
                leadtime_target_np = np.asarray(leadtime_target_td, dtype=coord_dtype)

            # If there is a leadtime dim, try to select the correct slice
            if leadtime_dim is not None:
                # Find nearest available leadtime in this chunk
                lt_values = ds_chunk[leadtime_dim].values

                logger.debug("   %s Leadtime values: %s", SOURCE_CTX, lt_values)
                unique_lt = np.unique(lt_values)

                # Target_np should be same dtype as unique_lt (timedelta or numeric)
                idx0 = np.argmin(np.abs(unique_lt - leadtime_target_np))
                center_lt = unique_lt[idx0]

                # Window around nearest center
                half_window = np.timedelta64(3, "D")
                low, high = center_lt - half_window, center_lt + half_window

                mask_lt = (lt_values >= low) & (lt_values <= high)
                idxs = np.where(mask_lt)[0]

                logger.debug(
                    "   %s Nearest leadtime center: %s, window: [%s, %s], inferred realizations %s",
                    SOURCE_CTX,
                    pd.to_timedelta(center_lt) if np.issubdtype(unique_lt.dtype, np.timedelta64) else center_lt,
                    pd.to_timedelta(low) if np.issubdtype(unique_lt.dtype, np.timedelta64) else low,
                    pd.to_timedelta(high) if np.issubdtype(unique_lt.dtype, np.timedelta64) else high,
                    idxs.size,
                )

                if idxs.size == 0:
                    raise ValueError(f"{SOURCE_CTX} No leadtimes found in the computed window")

                ds_chunk = ds_chunk.isel({leadtime_dim: idxs})
                logger.debug("   %s Chunk size after leadtime selection: %s", SOURCE_CTX, ds_chunk.sizes)

                # This crazy realization inference only for CMCC SPS4 ocean, it's very specific with leadtime dim being "time"
                if self.config.dataset == "seasonal-monthly-ocean":
                    ds_chunk = self._prepare_leadtime_and_infer_realizations("time", ds_chunk, start, end, (high, low), leadtime_target_np)
                    logger.debug("   %s Chunk size after realization inference: %s", SOURCE_CTX, ds_chunk.sizes)                    
            else:
                logger.debug("   %s Leadtime dim not found in fetched dataset. Continuing...", SOURCE_CTX)

            # Refresh realization and leadtime dim names
            time_dim = ds_chunk.earthml.guessed_dims.time
            realization_dim = ds_chunk.earthml.guessed_dims.realization
            leadtime_dim = ds_chunk.earthml.guessed_dims.leadtime
            leadtime_coord = ds_chunk.earthml.guessed_coords.leadtime

            # Make sure leadtime dim is named leadtime.name="leadtime"
            # At this point potentially after realization inference, leadtime_dim may be None
            if leadtime_dim is None:
                if leadtime_coord is None:
                    # Assign the coordinate to the requested target (ensures uniform coord across chunks)
                    ds_chunk = ds_chunk.expand_dims({leadtime.name: np.array([leadtime_target_np], dtype=coord_dtype)})
                else:
                    ds_chunk = ds_chunk.rename({leadtime_coord: leadtime.name})
            else:
                logger.debug(f"   {SOURCE_CTX} Leadtime values found: {ds_chunk[leadtime_dim].values}")
                # Move leadtime first to make bfill deterministic across the step axis
                ds_chunk = ds_chunk.transpose(leadtime_dim, ...)
                valid_per_leadtime = (~ds_chunk.to_array().isnull()).any(
                    dim=[d for d in ds_chunk.to_array().dims if d != leadtime_dim]
                )
                logger.debug(
                    "   %s Non-null leadtime slices: count=%s indices=%s",
                    SOURCE_CTX,
                    int(valid_per_leadtime.sum().item()),
                    np.where(valid_per_leadtime.values)[0],
                )
                # Since at most one is non-NaN, bfill then take leadtime=0 gives the only valid value
                # ds_chunk = ds_chunk.bfill(leadtime_dim).isel({leadtime_dim: 0})
                if leadtime_coord is None:
                    # Restore a length-1 leadtime dimension with the requested conceptual value
                    ds_chunk = ds_chunk.expand_dims({leadtime.name: np.array([leadtime_target_np], dtype=coord_dtype)})

            # Collapse monthly atmo forecast step -> one valid time and one data slice per sample.
            if self.config.dataset == "seasonal-monthly-single-levels" and leadtime_dim is not None:
                if "valid_time" not in ds_chunk.coords or leadtime_dim not in ds_chunk["valid_time"].dims:
                    raise ValueError(f"{SOURCE_CTX} Coord 'valid_time' with dim '{leadtime_dim}' not found in fetched dataset")

                vt = ds_chunk["valid_time"].transpose(leadtime_dim, ...)
                vt_vals = vt.values
                mask = ~pd.isna(vt_vals)

                has_valid = mask.any(axis=0)
                first_valid_idx = mask.argmax(axis=0)
                cols = np.where(has_valid)[0]

                vt_1d_vals = np.full(vt_vals.shape[1], np.datetime64("NaT"), dtype="datetime64[ns]")
                vt_1d_vals[cols] = vt_vals[first_valid_idx[cols], cols]

                if "time" in ds_chunk.coords:
                    ds_chunk = ds_chunk.assign_coords(reftime=("time", ds_chunk["time"].values))

                ds_chunk = ds_chunk.assign_coords(valid_time_1d=("time", vt_1d_vals))
                ds_chunk = ds_chunk.swap_dims({"time": "valid_time_1d"})
                ds_chunk = ds_chunk.drop_vars("time", errors="ignore")
                ds_chunk = ds_chunk.rename({"valid_time_1d": "time"})

                # Collapse data variables across step.
                for name in list(ds_chunk.data_vars):
                    da = ds_chunk[name]
                    if leadtime_dim in da.dims:
                        da = da.transpose(leadtime_dim, ...)
                        da = da.bfill(leadtime_dim).isel({leadtime_dim: 0}, drop=True)
                        ds_chunk[name] = da

                # Drop coordinates that still depend on step.
                coords_to_drop = [name for name, coord in ds_chunk.coords.items() if leadtime_dim in coord.dims]
                if coords_to_drop:
                    ds_chunk = ds_chunk.drop_vars(coords_to_drop, errors="ignore")

                # If step is still present as a scalar/size-1 coord, remove it.
                if leadtime_dim in ds_chunk.dims and ds_chunk.sizes.get(leadtime_dim, 0) == 1:
                    ds_chunk = ds_chunk.squeeze(leadtime_dim, drop=True)
                ds_chunk = ds_chunk.drop_vars(leadtime_dim, errors="ignore")


            logger.debug("   %s Size after all processing: %s", SOURCE_CTX, ds_chunk.sizes)
            logger.debug("   %s Coords after all processing: %s", SOURCE_CTX, ds_chunk.coords)

            xarray_concat_dim = ds_chunk.earthml.guessed_dims.time if not self.config.xarray_concat_dim else self.config.xarray_concat_dim
            logger.debug("   %s Discovered concat dim = time dim: %s", SOURCE_CTX, xarray_concat_dim)
            ds_chunks.append(ds_chunk)
            self._set_progress(
                progress,
                progress_task_id,
                total_chunks=total_chunks,
                chunk_position=chunk_index,
                stage="chunk complete",
                chunk_label=chunk_label,
            )

        return xarray_concat_dim, ds_chunks

    # TODO realization guessing is brittle, but I have no alternative for now
    def _prepare_leadtime_and_infer_realizations(
        self,
        orderable_leadtime_dim: str, # "time" in CDS CMCC SPS4 ocean monthly datasets
        ds_chunk: xr.Dataset,
        start: pd.Timedelta | datetime,
        end: pd.Timedelta | datetime,
        window: Sequence,
        leadtime_target: np.timedelta64,
    ):
        high, low = window

        leadtime_dim = ds_chunk.earthml.guessed_dims.leadtime
        lt = ds_chunk[leadtime_dim].values.copy() # store fore later reassignment

        # "time" is the date location of the leadtime, swap and sort
        ds_chunk = ds_chunk.swap_dims({leadtime_dim: orderable_leadtime_dim})
        # Use "time" to sort
        ds_chunk = ds_chunk.sortby(orderable_leadtime_dim)
        logger.debug(
            "   %s Realization inference, chunk coords after leadtime_dim=%s and orderable_leadtime_dim=%s swap: %s",
            SOURCE_CTX, leadtime_dim, orderable_leadtime_dim, ds_chunk.coords
        )

        # Infer member index from repeats within each time
        # This assumes that for each valid time we have the same number of stacked members.
        t = ds_chunk[orderable_leadtime_dim].values
        unique_t, counts = np.unique(t, return_counts=True)

        if counts.min() != counts.max():
            raise ValueError(f"{SOURCE_CTX} Unequal members per time: min={counts.min()} max={counts.max()} counts={dict(zip(unique_t, counts))}")

        n_realizations = int(counts[0])
        realization_index = np.concatenate([np.arange(n_realizations, dtype=np.int32) for _ in range(len(unique_t))])
        logger.debug("  %s Realization inference, number of realizations: %s, index: %s", SOURCE_CTX, n_realizations, realization_index)

        # Make leadtimes unique
        leadtime_per_date = lt.reshape(len(unique_t), n_realizations)[:, 0]
        mask_lt = (leadtime_per_date >= low) & (leadtime_per_date <= high)
        if np.all(mask_lt * leadtime_per_date):
            one_leadtime = leadtime_target
        else:
            raise ValueError(f"{SOURCE_CTX} More than one leadtime detected in tolerance window: {leadtime_per_date}")        

        # Add member coordinate and unstack to get dimensions ("time", "realization")
        ds_chunk = ds_chunk.assign_coords(realization=(orderable_leadtime_dim, realization_index))
        # logger.debug("  %s Realization inference, coord: %s", SOURCE_CTX, ds_chunk["realization"].values)

        # Avoid name collision
        ds_chunk = ds_chunk.rename_vars({orderable_leadtime_dim: "leadtime_date"})

        # MultiIndex
        ds_chunk = ds_chunk.set_index({orderable_leadtime_dim: ["leadtime_date", "realization"]}).unstack(orderable_leadtime_dim)

        # First remove any leftover leadtime
        if leadtime_dim in ds_chunk.coords or leadtime_dim in ds_chunk.data_vars:
            ds_chunk = ds_chunk.drop_vars(leadtime_dim)
        # Create real time axis
        ds_chunk = ds_chunk.rename({"leadtime_date": "time"})
        # Add new leadtime dim
        ds_chunk = ds_chunk.expand_dims({"leadtime": [one_leadtime]})
        ds_chunk = ds_chunk.assign_coords({
            "leadtime_date": ("time", leadtime_per_date),
            # "leadtime": ("leadtime", one_leadtime),
        })

        return ds_chunk

    def _month_splitter(self):
        all_months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        months_splitted = [
            [m for m in all_months[i:i+self.config.split_month] if m not in self.split_month_jump]
            for i in range(0, len(all_months), self.config.split_month)
        ]
        logger.debug("%s Months requested: %s", SOURCE_CTX, months_splitted)
        # Convert singletons to strings and clean up empty/None elements
        months_splitted = [chunk[0] if len(chunk) == 1 else chunk for chunk in months_splitted]
        months_splitted = [x for x in months_splitted if x]
        logger.debug("%s Months requested cleaned-up: %s", SOURCE_CTX, months_splitted)
        return months_splitted

    def _create_base_request_args_list(
        self,
        start: datetime,
        end: datetime,
        area: Sequence[int | float] ,
        split_req_dates: pd.DatetimeIndex | xr.CFTimeIndex | None,
    ) -> list[dict[str, Any]]:
        # Common request args
        request_args_d: dict[str, Any] = dict(variable=[v.longname for v in self.variables])
        if not self.config.select_area_after_request:
            request_args_d["area"] = area

        # Period request args
        request_time_args: dict[str, Any] = {}
        request_time_args_list = []

        if self.config.request_type == "hourly":
            if self.config.provider == "ecmwf-open-data":
                time_freq = generate_hours(self.data_selection.period.freq, 'int')
            else:
                time_freq = generate_hours(self.data_selection.period.freq)
            request_time_args['date'] = f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"
            request_time_args['time'] = time_freq
        elif self.config.request_type == "monthly":
            monthly_dates = xr.date_range(
                start=pd.Timestamp(start).to_period("M").to_timestamp(how="start"),
                end=pd.Timestamp(end).to_period("M").to_timestamp(how="start"),
                freq="MS",
            )
            request_time_args["year"] = sorted({d.strftime("%Y") for d in monthly_dates})
            if "month" not in self.config.request_extra_args:
                request_time_args["month"] = [d.strftime("%m") for d in monthly_dates]

        if split_req_dates is not None and self.config.request_type == "monthly":
            split_req_months = [m.strftime("%m") for m in split_req_dates]
            split_req_years = sorted({m.strftime("%Y") for m in split_req_dates})
            for m in self.months_splitted: # create multiple requests
                # logger.debug("  %s m:%s", SOURCE_CTX, m)
                request_time_args["year"] = split_req_years
                if "month" not in self.config.request_extra_args:
                    if isinstance(m, list):
                        month_req = [x for x in m if x in set(split_req_months)]
                        if not month_req:
                            continue
                    else:
                        if m in split_req_months:
                            month_req = m
                        else:
                            continue
                    request_time_args['month'] = month_req
                request_time_args_list.append(request_time_args | request_args_d)
        else: # only one request
            request_time_args_list.append(request_time_args | request_args_d)

        return request_time_args_list

    def _add_boundary_for_yearly_chunks(
        self,
        years: pd.DatetimeIndex | xr.CFTimeIndex,
        start: datetime,
        end: datetime,
    ) -> pd.DatetimeIndex | xr.CFTimeIndex:
        year_first, year_last = years[0], years[-1]
        if year_first > start: # comaprison may be broad
            new_years = xr.date_range(start, start, periods=1).append(years)
        else:
            new_years = years
        if year_last <= end:
            if self.config.request_type in ("daily", "hourly"):
                new_years = new_years.append(
                    xr.date_range(end + timedelta(days=1), end + timedelta(days=1), periods=1)
                ) #type: ignore
            elif self.config.request_type == "monthly":
                new_years = new_years.append(
                    xr.date_range(end + relativedelta(months=1), end + relativedelta(months=1), periods=1)
                ) #type: ignore
        return new_years #type: ignore

    def _get_data(self) -> xr.Dataset:
        n_missed = len(self.elements.missed) # can be nonzero if split_month_jump is non empty
        samples = [s for s in self.elements.samples if s not in self.elements.missed]
        logger.info("%s Samples: %s, missed: %s", SOURCE_CTX, len(samples), n_missed)

        logger.debug("%s Lead time: %s", SOURCE_CTX, self.config.leadtime)

        dates_d, years_d = {}, {}
        for v in self.variables:
            logger.debug("%s Variable: %s, period: %s - %s", SOURCE_CTX, v, self.start_d[v], self.end_d[v])
            # Optional hack to fix misaligned conventions by moving the request period (e.g. ORAS5 and SPS4 seadonal-monthly-ocean)
            # can be per-variable
            delta_hack = None
            if self.config.request_delta_hack:
                if isinstance(self.config.request_delta_hack, dict):
                    if v.name in self.config.request_delta_hack.keys():
                        delta_hack = self.config.request_delta_hack[v.name]
                elif isinstance(self.config.request_delta_hack, relativedelta):
                    delta_hack = self.config.request_delta_hack
                else:
                    delta_hack = relativedelta(months=0)
                    logger.debug("%s Invalid request_delta_hack %s, do not apply", SOURCE_CTX, self.config.request_delta_hack)
                if delta_hack is not None:
                    self.start_d[v] += delta_hack
                    self.end_d[v] += delta_hack
                    logger.debug(
                        "%s Anticipate %s %s request of %s: %s - %s",
                        SOURCE_CTX,
                        v.name,
                        self.config.dataset,
                        delta_hack,
                        self.start_d[v],
                        self.end_d[v],
                    )

            dates_d[v] = f"{self.start_d[v].strftime('%Y-%m-%d')}/{self.end_d[v].strftime('%Y-%m-%d')}"
            years_d[v] = self._add_boundary_for_yearly_chunks(
                xr.date_range(
                    start=datetime(self.start_d[v].year, 1, 1),
                    end=datetime(self.end_d[v].year, 1, 1),
                    freq="YS",
                ),
                self.start_d[v],
                self.end_d[v],
            )
            if len(years_d[v]) == 0:
                raise ValueError(
                    f"{SOURCE_CTX} Years calculated from date range for var {v.name} are empty for start={self.start_d[v]} end={self.end_d[v]}"
                )
            logger.debug("%s Requested split-by-year ranges for var %s: %s", SOURCE_CTX, v.name, list(years_d[v]))
        
        self.months_splitted = self._month_splitter()

        area: list[int | float] = [
            self.data_selection.region.lat[0],
            self.data_selection.region.lon[0],
            self.data_selection.region.lat[1],
            self.data_selection.region.lon[1]
        ]

        for v in self.variables:
            start, end = self.start_d[v], self.end_d[v]
            years = years_d[v]
            leadtime_req = self.leadtime_d[v]["request"] # we request all leadtimes at once to optimize caching
            leadtime_var = self.leadtime_d[v]["dataset"]

            xarray_concat_dim = None
            split_req_dates = None
            ds_vars = []
            # Single year requests
            if (
                self.config.split_request and (
                    end - start > pd.to_timedelta('365 days') or
                    end.year == start.year+1
                )
            ):
                logger.debug("%s Single year requests for %s", SOURCE_CTX, v)
                datasets: Sequence[xr.Dataset] = []
                concat_dims: list[str | None] = []
                year_chunk_requests: list[tuple[pd.Timestamp, pd.Timestamp, list[dict[str, Any]]]] = []

                for y1, y2 in zip(years[:-1], years[1:]):
                    raw_chunk_start = max(pd.Timestamp(y1), pd.Timestamp(start))

                    if self.config.request_type in ("daily", "hourly"):
                        chunk_start = raw_chunk_start
                        chunk_end = min(pd.Timestamp(y2) - timedelta(days=1), pd.Timestamp(end))
                        split_req_dates = None

                    elif self.config.request_type == "monthly":
                        # Canonicalize monthly bounds to month starts before splitting.
                        start_ms = pd.Timestamp(start).to_period("M").to_timestamp(how="start")
                        end_ms = pd.Timestamp(end).to_period("M").to_timestamp(how="start")
                        y1_ms = pd.Timestamp(y1).to_period("M").to_timestamp(how="start")
                        y2_ms = pd.Timestamp(y2).to_period("M").to_timestamp(how="start")

                        chunk_start = max(y1_ms, start_ms)
                        chunk_end = min(y2_ms - relativedelta(months=1), end_ms)

                        if chunk_start > chunk_end:
                            continue

                        split_req_dates = xr.date_range(start=chunk_start, end=chunk_end, freq="MS")
                        if len(split_req_dates) == 0:
                            continue

                    else:
                        raise ValueError(f"{SOURCE_CTX} Unsupported earthkit request type {self.config.request_type}")

                    if chunk_start > chunk_end:
                        continue

                    request_args_list = self._create_base_request_args_list(
                        chunk_start, chunk_end, area, split_req_dates
                    )
                    if request_args_list:
                        year_chunk_requests.append((chunk_start, chunk_end, request_args_list))
                    else:
                        logger.warning("%s Cannot fetch request. Continuing...", SOURCE_CTX)

                total_chunks = sum(len(reqs) for _, _, reqs in year_chunk_requests)

                offset = 0
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                ) as prog:
                    logger.info("%s Check request status: https://cds.climate.copernicus.eu/requests?tab=all", SOURCE_CTX)
                    task = prog.add_task(
                        self._progress_task_description(v.longname, dates_d[v]),
                        total=total_chunks,
                    )

                    for chunk_start, chunk_end, request_args_list in year_chunk_requests:
                        c_dim, ds_chunks = self._fetch_chunks(
                            request_args_list=request_args_list,
                            start=chunk_start,
                            end=chunk_end,
                            leadtime=leadtime_var,
                            request_other_args=self.config.request_extra_args,
                            total_chunks=total_chunks,
                            chunk_offset=offset,
                            progress=prog,
                            progress_task_id=task,
                        )
                        offset += len(request_args_list)
                        datasets.extend(ds_chunks)
                        concat_dims.append(c_dim)

                    if None in concat_dims:
                        raise ValueError(f"{SOURCE_CTX} Not all chunks have a valid time dim for concatenation: {concat_dims}")

                    if len(set(concat_dims)) == 1:
                        xarray_concat_dim = concat_dims[0]
                    else:
                        raise ValueError(f"{SOURCE_CTX} Different time dim for concatenation: {concat_dims}")

            # Multiple years single request
            else:
                logger.debug("%s Multiple years requests for %s", SOURCE_CTX, v)
                request_args_list = self._create_base_request_args_list(start, end, area, split_req_dates)
                
                if request_args_list:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                    ) as prog:
                        logger.info("%s Check request status: https://cds.climate.copernicus.eu/requests?tab=all", SOURCE_CTX)
                        task = prog.add_task(
                            self._progress_task_description(v.longname, dates_d[v]),
                            total=len(request_args_list),
                        )
                        xarray_concat_dim, datasets = self._fetch_chunks(
                            request_args_list=request_args_list,
                            start=start,
                            end=end,
                            leadtime=leadtime_var, # used to filter leadtime in actual dataset
                            request_other_args=self.config.request_extra_args,
                            progress=prog,
                            progress_task_id=task,
                        )
                else:
                    logger.warning("%s Cannot fetch request. Continuing...", SOURCE_CTX)

            # Guard for empy datasets and no concat dim
            if not datasets or xarray_concat_dim is None:
                raise ValueError(
                    f"{SOURCE_CTX} No datasets fetched for {self.source_name}. "
                    f"request_type={self.config.request_type}, start={start}, end={end}"
                )

            # Combine all datasets
            logger.debug("%s Combine split-request datasets of length for var %s %s", SOURCE_CTX, v.name, len(datasets))
            ds_single_var = xr.concat(datasets, dim=xarray_concat_dim, **self.config.xarray_concat_extra_args)
            logger.debug("%s Datasets combined length for var %s: %s", SOURCE_CTX, v.name, len(ds_single_var[xarray_concat_dim].values))
            ds_vars.append(ds_single_var)

        # Merge
        ds_all = xr.merge(ds_vars)
        logger.debug("%s Merged dataset for vars %s", SOURCE_CTX, [v.name for v in self.variables])

        # Drop unused variables
        ds_all = ds_all.drop_vars([v for v in ds_all.data_vars if v not in self.var_name_list])

        # Ensure concat dim is datetime64[ns]
        ds_all = xr.decode_cf(ds_all)
        ds_all = ds_all.assign_coords({xarray_concat_dim: ds_all[xarray_concat_dim].astype("datetime64[ns]")})
        logger.debug("%s Time values before reassignment: %s", SOURCE_CTX, ds_all[xarray_concat_dim].values)

        # Snapping
        t = pd.to_datetime(ds_all[xarray_concat_dim].values)
        # Snap some datasets timesteps to month-start of the PREVIOUS month (00:00 of the 1st)
        if self.config.snap_time_index == "previous":
            k = -1
        # Snap some datasets timesteps to month-start of the NEXT month (00:00 of the 1st) (e.g. ORAS5 and ocean SPS4 weird convention that half-month -> next month)
        if self.config.snap_time_index == "next":
            k = 1
        # Snap some datasets timesteps to month-start of the SAME month (00:00 of the 1st) (e.g. ERA5 is at 12:00 start-of-month)
        if self.config.snap_time_index == "same":
            k = 0
        if self.config.snap_time_index == "nearest":
        # Snap some datasets timesteps to the NEAREST month month-start (00:00 of the 1st) (e.g. atmo SPS4, with 'valid_time')
            month_start = t.to_period("M").to_timestamp(how="start")
            next_month_start = (t.to_period("M") + 1).to_timestamp(how="start")
            t_snapped = month_start.where(
                (t - month_start) <= (next_month_start - t),
                next_month_start,
            )
        else:
            t_snapped = (t.to_period("M") + k).to_timestamp(how="start")
        
        ds_all = ds_all.assign_coords({xarray_concat_dim: (xarray_concat_dim, t_snapped.values)})

        logger.debug("%s Datasets before dropping missing samples: %s", SOURCE_CTX, len(ds_all[xarray_concat_dim].values))
        logger.debug("%s Time values after reassignment: %s", SOURCE_CTX, ds_all[xarray_concat_dim].values)

        n_missed = len(self.elements.missed) # recompute after fetching
        n_actual_samples = len(ds_all[xarray_concat_dim].values)
        n_all_samples = len(self.date_range)
        assert n_missed + n_actual_samples == n_all_samples, f"{SOURCE_CTX} number of samples obtained + missed is different from number of all samples ({n_actual_samples}+{n_missed} =/= {n_all_samples})"
        logger.debug("%s Missed: %s", SOURCE_CTX, self.elements.missed)
        # Drop missing samples
        xarray_concat_dim = ds_all.earthml.guessed_dims.time if not self.config.xarray_concat_dim else self.config.xarray_concat_dim
        if self.elements.missed:
            ds_all = ds_all.drop_sel({xarray_concat_dim: list(self.elements.missed)}, errors='ignore')

        logger.debug("%s Timesteps after dropping missing samples: %s", SOURCE_CTX, len(ds_all[xarray_concat_dim].values))

        # import sys
        # sys.exit("Stopping here on purpose")

        return ds_all
