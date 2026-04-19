import datetime
from typing import Literal, Any
from dataclasses import dataclass, field
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading

import re, dask
from heapq import nlargest
from pathlib import Path

import hashlib
import json

from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr

from dask.utils import SerializableLock

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

from ..base import Leadtime
from ..logging import get_console, get_logger

from .dataclasses import DataSource, Sample, RegridConfig
from .utils import retry_fetch_after_hdf_err
from .xarray_local import MFXarrayLocalSource
from ._preprocess import preprocess_mfdataset


REALIZATION_RE = re.compile(r"_r(\d+)")
logger = get_logger(__name__)
SOURCE_CTX = "[source juno-local]"


@dataclass
class JunoLocalSourceFileNameConfig:
    file_path_header                    : str = ""
    file_path_suffix                    : str = ""
    file_path_date_format               : str = ""
    file_header                         : str = ""
    file_suffix                         : str = ""
    file_date_format                    : str = "%Y%m%d"
    both_data_and_previous_date_in_file : bool = True
    date_order                          : Literal["previous_current", "current_previous"] = "previous_current"
    date_separator                      : str = ""
    realizations                        : int | Literal["all"] = 1
    minus_timedelta                     : timedelta | None = None
    plus_timedelta                      : timedelta | None = None


@dataclass
class JunoLocalSourcePathConfig:
    start_date                  : str | pd.Timestamp
    end_date                    : str | pd.Timestamp | None = None
    root_path                   : str | Path | None = None
    file_name_config            : JunoLocalSourceFileNameConfig | None = None


@dataclass
class JunoLocalSourceConfig:
    leadtime                    : Leadtime
    root_path                   : str | Path
    engine                      : str
    regrid_config               : RegridConfig
    per_sample_regrid           : bool = False
    file_name_config            : JunoLocalSourceFileNameConfig | None = None
    path_configs                : list[JunoLocalSourcePathConfig] = field(default_factory=list)
    select_area_after_request   : bool = True
    cfgrib_idx_path             : str | Path = ""
    file_open_workers           : int | None = 1
    chunk_option                : Any = "auto"


class JunoLocalSource(MFXarrayLocalSource):
    """
    Collect Juno local data for the given data selection.
    We do not need "shifted" in TimeRange because we use leadtime to select
    the correct input relative to the date range.
    """
    def __init__(
        self,
        datasource: DataSource,
        config: JunoLocalSourceConfig
    ):
        super().__init__(datasource, config)
        self.config = config
        self.engine = config.engine

        self.leadtime = config.leadtime
        self.elements = self._get_data_filenames(self.leadtime, config.file_name_config)

        self.select_area_after_request = self.config.select_area_after_request

        self.per_sample_regrid = bool(config.per_sample_regrid)
        self.sample_regrid_resolution = config.regrid_config.regrid_resolution
        self.regrid_vars = config.regrid_config.regrid_vars if config.regrid_config.regrid_vars is not None else self.var_name_list
        self.regrid_resolution = None if self.per_sample_regrid else self.sample_regrid_resolution

        self.cfgrib_idx_path = str(config.cfgrib_idx_path)
        if self.engine == "cfgrib" and self.cfgrib_idx_path:
            Path(self.cfgrib_idx_path).mkdir(parents=True, exist_ok=True)

        self.file_open_workers = self._resolve_file_open_workers(config.file_open_workers)
        self._cfgrib_locks: dict[str, SerializableLock] = {}
        self._cfgrib_locks_guard = threading.Lock()


    @staticmethod
    def _resolve_file_open_workers(file_open_workers: int | None) -> int:
        if file_open_workers is None:
            return max(1, multiprocessing.cpu_count())
        if file_open_workers < 1:
            raise ValueError("file_open_workers must be >= 1 or None")
        return file_open_workers


    @staticmethod
    def _make_cfgrib_index_key(path: Path, filter_by_keys: dict | None = None) -> str:
        """Create unique index key for grib indexing"""
        try:
            stat = path.stat()
            file_sig = {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except FileNotFoundError:
            file_sig = {
                "path": str(path.resolve()),
                "size": None,
                "mtime_ns": None,
            }

        key_data = {
            "file": file_sig,
            "filter_by_keys": filter_by_keys or {},
        }

        raw = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(raw.encode()).hexdigest()


    def _get_cfgrib_lock(self, path: Path) -> SerializableLock:
        lock_key = str(path.resolve())
        with self._cfgrib_locks_guard:
            lock = self._cfgrib_locks.get(lock_key)
            if lock is None:
                lock = SerializableLock()
                self._cfgrib_locks[lock_key] = lock
                logger.debug("%s Created cfgrib lock for %s", SOURCE_CTX, lock_key)
        return lock


    def _open_one_cfgrib(self, path: Path, var_name: str, date):
        filter_by_keys = {"cfVarName": var_name}
        indexpath_key = self._make_cfgrib_index_key(path, filter_by_keys)
        indexpath = str(Path(self.cfgrib_idx_path) / f"{indexpath_key}.idx")
        lock = self._get_cfgrib_lock(path)

        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={
                "indexpath": indexpath,
                "filter_by_keys": filter_by_keys,
            },
            decode_cf=True,
            decode_timedelta=True,
            lock=lock,
            chunks="auto",
        )

        return preprocess_mfdataset(
            ds,
            data=self.data_selection,
            var_name=var_name,
            date=date,
        )

    @staticmethod
    def _latest_matching_files(data_path: Path, pattern: str, realizations: int | Literal["all"]) -> list[Path]:
        """
        Return matching files sorted by newest mtime first.
        Preserves the original behavior:
        - only files
        - sorted by st_mtime descending
        - if realizations != 'all', keep only the newest `realizations`
        """
        matches = []
        for p in data_path.glob(pattern):
            if p.is_file():
                try:
                    mtime = p.stat().st_mtime
                except FileNotFoundError:
                    # File vanished between glob and stat; equivalent to not found
                    continue
                matches.append((mtime, p))

        if not matches:
            return []

        if realizations == "all":
            matches.sort(key=lambda t: t[0], reverse=True)
            return [p for _, p in matches]

        # Newest files only, newest-first order
        top = nlargest(realizations, matches, key=lambda t: t[0])
        top.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in top]

    @staticmethod
    def _realization_key(p: Path) -> int:
        m = REALIZATION_RE.search(p.name)
        if not m:
            raise ValueError(f"Could not parse realization from filename: {p.name}")
        return int(m.group(1))

    @staticmethod
    def _build_data_path(
        config: JunoLocalSourceFileNameConfig,
        root_path: Path,
        date: datetime,
    ) -> Path:
        date_str = date.strftime(config.file_path_date_format)
        inner_path = f"{config.file_path_header}{date_str}{config.file_path_suffix}"
        return root_path.joinpath(inner_path)

    @staticmethod
    def _build_data_glob(
        *,
        config: JunoLocalSourceFileNameConfig,
        previous_date,
        current_date,
    ) -> str:
        prev_str = previous_date.strftime(config.file_date_format)
        date_str = current_date.strftime(config.file_date_format)

        if config.both_data_and_previous_date_in_file:
            if config.date_order == "current_previous":
                return f"{config.file_header}{date_str}{config.date_separator}{prev_str}{config.file_suffix}"
            return f"{config.file_header}{prev_str}{config.date_separator}{date_str}{config.file_suffix}"

        return f"{config.file_header}{prev_str}{config.file_suffix}"

    @staticmethod
    def _as_timestamp(value) -> pd.Timestamp | None:
        return None if value is None else pd.Timestamp(value)

    def _resolve_path_config(
        self,
        previous_date,
    ) -> tuple[Path, JunoLocalSourceFileNameConfig]:
        ts = pd.Timestamp(previous_date)

        for path_config in self.config.path_configs:
            start_date = self._as_timestamp(path_config.start_date)
            end_date = self._as_timestamp(path_config.end_date)
            if ts < start_date:
                continue
            if end_date is not None and ts > end_date:
                continue

            root_path = self.path if path_config.root_path is None else Path(path_config.root_path)
            file_name_config = (
                self.config.file_name_config
                if path_config.file_name_config is None
                else path_config.file_name_config
            )
            return root_path, file_name_config

        return self.path, self.config.file_name_config

    def _open_sample_dataset(
        self,
        sample: list[Path],
        date,
        *,
        realization_concat_dim: str,
        selected_vars: list,
        has_shifted_samples: bool,
        lock,
    ) -> xr.Dataset | None:
        if len(sample) > 1:
            sample = sorted(sample, key=self._realization_key)

        common_args = {
            "combine": "nested",
            "coords": "minimal" if has_shifted_samples else "different",
            "compat": "override" if has_shifted_samples else "no_conflicts",
            "engine": self.engine,
            "chunks": self.config.chunk_option,
            "parallel": True,
            "decode_timedelta": True,
            "backend_kwargs": {},
            "preprocess": partial(preprocess_mfdataset, data=self.data_selection),
            "decode_cf": True,
            "errors": "warn",
            "lock": lock,
            "concat_dim": realization_concat_dim,
            "paths": sample,
        }

        is_var_list = isinstance(self.data_selection.variable, list)

        if is_var_list:
            var_ds_list = []

            for var in selected_vars:
                if self.engine == "cfgrib":
                    per_file = []
                    for path in sample:
                        def _open_one(path=path, var_name=var.name, date=date):
                            return self._open_one_cfgrib(path, var_name, date)

                        ds_one = retry_fetch_after_hdf_err(
                            _open_one,
                            error_re=r"Unspecified error in H5DSget_num_scales.*",
                        )
                        per_file.append(ds_one)

                    ds_var = xr.concat(
                        per_file,
                        dim=realization_concat_dim,
                        coords="minimal",
                        compat="override",
                        combine_attrs="drop_conflicts",
                    )
                    var_ds_list.append(ds_var)
                else:
                    var_args = dict(common_args)
                    var_args["preprocess"] = partial(
                        preprocess_mfdataset,
                        data=self.data_selection,
                        var_name=var.name,
                        date=date,
                    )

                    def _open_mfdataset_local(local_args=var_args):
                        return xr.open_mfdataset(**local_args)

                    var_ds = retry_fetch_after_hdf_err(
                        _open_mfdataset_local,
                        error_re=r"Unspecified error in H5DSget_num_scales.*",
                    )
                    var_ds_list.append(var_ds)

            ds_sample = xr.merge(var_ds_list, compat="no_conflicts", combine_attrs="no_conflicts")

            if realization_concat_dim in ds_sample.coords:
                ds_sample = ds_sample.assign_coords(
                    {realization_concat_dim: np.asarray(ds_sample[realization_concat_dim].values)}
                )
        else:
            var = selected_vars[0]

            if self.engine == "cfgrib":
                per_file = []
                for path in sample:
                    def _open_one(path=path, var_name=var.name, date=date):
                        return self._open_one_cfgrib(path, var_name, date)

                    ds_one = retry_fetch_after_hdf_err(
                        _open_one,
                        error_re=r"Unspecified error in H5DSget_num_scales.*",
                    )
                    per_file.append(ds_one)

                ds_sample = xr.concat(
                    per_file,
                    dim=realization_concat_dim,
                    coords="minimal",
                    compat="override",
                    combine_attrs="drop_conflicts",
                )
            else:
                args = dict(common_args)
                args["preprocess"] = partial(
                    preprocess_mfdataset,
                    data=self.data_selection,
                    var_name=var.name,
                    date=date,
                )

                def _open_mfdataset_local(local_args=args):
                    return xr.open_mfdataset(**local_args)

                ds_sample = retry_fetch_after_hdf_err(
                    _open_mfdataset_local,
                    error_re=r"Unspecified error in H5DSget_num_scales.*",
                )

            if realization_concat_dim in ds_sample.coords:
                ds_sample = ds_sample.assign_coords(
                    {realization_concat_dim: np.asarray(ds_sample[realization_concat_dim].values)}
                )
            if realization_concat_dim in ds_sample.dims:
                ds_sample = ds_sample.assign_coords(
                    {realization_concat_dim: ds_sample[realization_concat_dim]}
                )

        return self._normalize_sample_dataset(ds_sample, date)

    def _normalize_sample_dataset(self, ds_sample: xr.Dataset, date) -> xr.Dataset | None:
        if "_has_var" not in ds_sample:
            raise ValueError(f"{date}: ds_sample has no _has_var flag")

        if not bool(ds_sample["_has_var"].any().item()):
            return None

        lon_name = ds_sample.earthml.guessed_coords.longitude
        lat_name = ds_sample.earthml.guessed_coords.latitude

        if lon_name is None or lat_name is None:
            return None

        lon_da = ds_sample[lon_name]
        lon = np.asarray(lon_da.values, dtype=np.float64)
        lon = ((lon + 180.0) % 360.0) - 180.0
        lon = np.round(lon, 10)
        lon = np.where(np.isclose(lon, 180.0, atol=1e-8), -180.0, lon)

        # Reassign longitude with explicit dims
        ds_sample = ds_sample.assign_coords({
            lon_name: (lon_da.dims, lon)
        })

        # 1D regular grid: keep old behavior
        if lon_da.ndim == 1:
            ds_sample = ds_sample.sortby(lon_name)

            vals = np.asarray(ds_sample[lon_name].values)
            _, idx = np.unique(vals, return_index=True)
            ds_sample = ds_sample.isel({lon_name: np.sort(idx)})

        # 2D curvilinear grid: cannot sort/dedup by lon coordinate this way
        elif lon_da.ndim == 2:
            pass

        else:
            raise ValueError(
                f"{date}: unsupported longitude coord {lon_name!r} with ndim={lon_da.ndim}"
            )

        if self.per_sample_regrid and self.sample_regrid_resolution is not None:
            ds_sample = ds_sample.earthml.regrid_to_rectilinear(
                region=self.data_selection.region,
                resolution=self.sample_regrid_resolution,
                vars_to_regrid=self.regrid_vars,
                silent=True,
            )

        return ds_sample

    def _get_data_filenames(
        self,
        leadtime: Leadtime,
        config: JunoLocalSourceFileNameConfig,
    ) -> Sample:
        """Get the data filenames for the given data selection."""

        s = Sample(extra={"plus_samples": [], "minus_samples": []})
        assert config.realizations == "all" or config.realizations > 0

        for date in self.date_range:
            previous_date = date - leadtime.to_timedelta()
            root_path, date_config = self._resolve_path_config(previous_date)
            data_path = self._build_data_path(
                config=date_config,
                root_path=root_path,
                date=previous_date,
            )
            logger.debug("%s Data path constructed for date %s: %s", SOURCE_CTX, date, data_path)

            realizations = date_config.realizations
            minus_td = date_config.minus_timedelta
            plus_td = date_config.plus_timedelta
            data_glob = self._build_data_glob(
                config=date_config,
                previous_date=previous_date,
                current_date=date,
            )
            logger.debug("%s Data glob constructed for date %s: %s", SOURCE_CTX, date, data_glob)

            files_exact = self._latest_matching_files(data_path, data_glob, realizations)

            if files_exact:
                s.samples[date] = files_exact
                continue

            found = False

            if minus_td is not None:
                test_date = date - minus_td
                test_glob = self._build_data_glob(
                    config=date_config,
                    previous_date=previous_date,
                    current_date=test_date,
                )

                test_files = self._latest_matching_files(data_path, test_glob, realizations)
                if test_files:
                    s.samples[date] = test_files
                    s.extra["minus_samples"].append(date)
                    found = True

            if not found and plus_td is not None:
                test_date = date + plus_td
                test_glob = self._build_data_glob(
                    config=date_config,
                    previous_date=previous_date,
                    current_date=test_date,
                )

                test_files = self._latest_matching_files(data_path, test_glob, realizations)
                if test_files:
                    s.samples[date] = test_files
                    s.extra["plus_samples"].append(date)
                    found = True

            if not found:
                logger.warning("%s Missed sample (local filename not found): %s", SOURCE_CTX, date)
                s.missed.add(date)

        return s


    def _get_data(self) -> xr.Dataset:
        samples = [s for date, s in self.elements.samples.items() if date not in self.elements.missed]
        assert len(samples) > 0, "No samples obtained."
        dates = [date for date in self.elements.samples.keys() if date not in self.elements.missed]

        minus_samples = self.elements.extra["minus_samples"]
        plus_samples = self.elements.extra["plus_samples"]
        has_shifted_samples = bool(minus_samples or plus_samples)

        logger.info(
            "%s Juno local file samples: %s, minus: %s, plus: %s, missed: %s",
            SOURCE_CTX,
            len(samples),
            len(minus_samples),
            len(plus_samples),
            len(self.elements.missed),
        )

        samples_d: dict[pd.Timestamp, xr.Dataset] = {}
        lock = SerializableLock()

        common_args = {
            "backend_kwargs": {},
        }

        # Prepare probe backend kwargs consistent with common_args
        first_var = (
            self.data_selection.variable[0].name
            if isinstance(self.data_selection.variable, list)
            else self.data_selection.variable.name
        )

        probe_backend_kwargs = dict(common_args["backend_kwargs"])
        if self.engine == "cfgrib":
            probe_backend_kwargs.update({
                "indexpath": "",
                "filter_by_keys": {"cfVarName": first_var},
            })

        # Safely guess concat_dim from first file
        with xr.open_dataset(
            samples[0][0],
            engine=self.engine,
            backend_kwargs=probe_backend_kwargs,
            lock=lock,
        ) as ds0:
            realization_concat_dim = ds0.earthml.guessed_dims.realization or "realization"

        common_args["concat_dim"] = realization_concat_dim

        is_var_list = isinstance(self.data_selection.variable, list)
        selected_vars = self.data_selection.variable if is_var_list else [self.data_selection.variable]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=get_console(),
        ) as prog:
            task = prog.add_task("Opening local Juno samples", total=len(samples))

            def _load_for_date(sample, date):
                assert isinstance(sample, list), f"Sample should be a list but it is {type(sample)}"
                return self._open_sample_dataset(
                    sample,
                    date,
                    realization_concat_dim=realization_concat_dim,
                    selected_vars=selected_vars,
                    has_shifted_samples=has_shifted_samples,
                    lock=lock,
                )

            if self.file_open_workers == 1:
                for sample, date in zip(samples, dates):
                    ds_sample = _load_for_date(sample, date)
                    if ds_sample is None:
                        self.elements.missed.add(date)
                    else:
                        samples_d[date] = ds_sample
                    logger.debug(f"{SOURCE_CTX} Juno local, vars in sample [{date}]: {ds_sample.data_vars}")
                    prog.advance(task)
            else:
                workers = min(self.file_open_workers, len(samples))
                logger.info("%s Opening Juno local samples with %s worker threads", SOURCE_CTX, workers)
                future_to_date = {}

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for sample, date in zip(samples, dates):
                        future = pool.submit(_load_for_date, sample, date)
                        future_to_date[future] = date

                    for future in as_completed(future_to_date):
                        date = future_to_date[future]
                        ds_sample = future.result()
                        if ds_sample is None:
                            self.elements.missed.add(date)
                        else:
                            samples_d[date] = ds_sample
                        prog.advance(task)
                        logger.debug(f"{SOURCE_CTX} Juno local, vars in sample [{date}]: {ds_sample.data_vars}")

        dbg_sample_k = self.date_range[10]
        logger.debug(f"{SOURCE_CTX} Juno local, size of ds[{dbg_sample_k}] after open_mfdataset: {samples_d[dbg_sample_k].sizes}")

        # Infer resolution from first valid sample
        valid_dates = sorted(samples_d)
        ref_date = valid_dates[0]
        ref_lon_name = samples_d[ref_date].earthml.guessed_coords.longitude
        ref_lon_da = samples_d[ref_date][ref_lon_name]
        ref_lon = np.asarray(ref_lon_da.values, dtype=np.float64)
        ref_lon = np.round(ref_lon, 10)

        if ref_lon_da.ndim == 1:
            ref_lon = np.unique(ref_lon)  # remove exact duplicates on regular grids
            logger.info("%s Juno local inferred reference 1D lon size: %s", SOURCE_CTX, ref_lon.size)
        elif ref_lon_da.ndim == 2:
            logger.info("%s Juno local inferred reference 2D lon shape: %s", SOURCE_CTX, ref_lon.shape)
        else:
            raise ValueError(
                f"{ref_date}: unsupported reference longitude coord {ref_lon_name!r} "
                f"with ndim={ref_lon_da.ndim}"
            )

        # Enforce same lon convention to all samples
        for date, ds_sample in samples_d.items():
            lon_name = ds_sample.earthml.guessed_coords.longitude
            lon_da = ds_sample[lon_name]
            vals = np.asarray(lon_da.values, dtype=np.float64)
            vals = np.round(vals, 10)

            if lon_da.ndim != ref_lon_da.ndim:
                raise ValueError(
                    f"{date}: longitude ndim mismatch {lon_da.ndim} vs ref {ref_lon_da.ndim}"
                )

            if lon_da.ndim == 1:
                if vals.size != ref_lon.size:
                    raise ValueError(
                        f"{date}: longitude size mismatch {vals.size} vs ref {ref_lon.size}"
                    )

                if not np.allclose(vals, ref_lon, atol=1e-8, rtol=0.0):
                    diff = np.max(np.abs(vals - ref_lon))
                    raise ValueError(
                        f"{date}: longitude grid differs from reference, max abs diff={diff}"
                    )

                samples_d[date] = ds_sample.assign_coords({lon_name: ref_lon})
                continue

            if vals.shape != ref_lon.shape:
                raise ValueError(
                    f"{date}: longitude shape mismatch {vals.shape} vs ref {ref_lon.shape}"
                )

            if not np.allclose(vals, ref_lon, atol=1e-8, rtol=0.0):
                diff = np.max(np.abs(vals - ref_lon))
                raise ValueError(
                    f"{date}: longitude grid differs from reference, max abs diff={diff}"
                )

            samples_d[date] = ds_sample.assign_coords({lon_name: (lon_da.dims, ref_lon)})

        # Batch validity checks instead of one compute() per sample
        validity_tasks = []
        validity_dates = []

        for date, ds in samples_d.items():
            time_dim = ds.earthml.guessed_dims.time
            dims = tuple(
                d for d in (realization_concat_dim, time_dim)
                if d is not None and d in ds["_has_var"].dims
            )
            validity_tasks.append(ds["_has_var"].any(dim=dims))
            validity_dates.append(date)

        validity_values = dask.compute(*validity_tasks) if validity_tasks else ()

        samples_len = []
        missing_samples = []

        for date, ds, has_any in zip(validity_dates, (samples_d[d] for d in validity_dates), validity_values):
            ok = bool(has_any.item())
            if ok:
                samples_len.append(ds.sizes.get(realization_concat_dim, 1))
            else:
                # Store dates with no valid realizations
                missing_samples.append(date)

        if not samples_len:
            raise ValueError(
                f"No valid samples found for {self.source_name}. "
                f"Total candidates={len(samples_d)}, missing_samples={len(missing_samples)}"
            )

        # Use only min_R realizations per sample
        min_R = min(samples_len)

        for date, ds in samples_d.items():
            if realization_concat_dim in ds.dims:
                dsR = ds.isel({realization_concat_dim: slice(0, min_R)})
                R = dsR.sizes[realization_concat_dim]
                samples_d[date] = dsR.assign_coords({realization_concat_dim: np.arange(R)})

        self.elements.missed.update(pd.to_datetime(missing_samples).to_pydatetime().tolist())

        # Concatenate
        times = np.array(
            [d for d in sorted(samples_d.keys()) if d not in self.elements.missed],
            dtype="datetime64[ns]",
        )
        objs = [samples_d[d] for d in sorted(samples_d) if d not in self.elements.missed]

        if not objs:
            raise ValueError(
                f"No datasets left to concatenate for {self.source_name} after removing missed samples."
            )

        concat_time_dim = (
            samples_d[ref_date].earthml.guessed_dims.time
            or samples_d[ref_date].earthml.guessed_coords.time
            or "time"
        )
        # TODO add progress bar to concat too?
        ds_all = xr.concat(
            objs=objs,
            dim=xr.IndexVariable(concat_time_dim, times),
            coords="minimal",
            compat="override",
            join="outer",
            combine_attrs="drop_conflicts",
        )
        # progress(ds_all) # show progress bar (concat lazy, not working)
        logger.info("%s Juno local, size of ds after concat: %s", ds_all.sizes, SOURCE_CTX)

        return ds_all
