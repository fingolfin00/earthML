import datetime
from typing import Literal, Any
from dataclasses import dataclass, field
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading
import time

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
    per_product_regrid          : bool = False
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
        self.elements = self._get_data_filenames(config.file_name_config)

        self.select_area_after_request = self.config.select_area_after_request

        self.per_sample_regrid = bool(config.per_sample_regrid)
        self.per_product_regrid = bool(config.per_product_regrid)
        if self.per_sample_regrid and self.per_product_regrid:
            raise ValueError("per_sample_regrid and per_product_regrid cannot both be True")
        self.sample_regrid_resolution = config.regrid_config.regrid_resolution
        self.regrid_vars = config.regrid_config.regrid_vars if config.regrid_config.regrid_vars is not None else self.var_name_list
        self.regrid_resolution = None if (self.per_sample_regrid or self.per_product_regrid) else self.sample_regrid_resolution

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
        t0 = time.perf_counter()
        filter_by_keys = {"cfVarName": var_name}
        indexpath_key = self._make_cfgrib_index_key(path, filter_by_keys)
        indexpath = str(Path(self.cfgrib_idx_path) / f"{indexpath_key}.idx")
        lock = self._get_cfgrib_lock(path)

        t_open = time.perf_counter()
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
        t_preprocess = time.perf_counter()

        out = preprocess_mfdataset(
            ds,
            data=self.data_selection,
            var_name=var_name,
            date=date,
        )
        t_done = time.perf_counter()
        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
        except FileNotFoundError:
            file_size_mb = float("nan")
        logger.debug(
            "%s cfgrib timings date=%s var=%s file=%s size_mb=%.1f lock+setup=%.3fs open=%.3fs preprocess=%.3fs total=%.3fs",
            SOURCE_CTX,
            date,
            var_name,
            path,
            file_size_mb,
            t_open - t0,
            t_preprocess - t_open,
            t_done - t_preprocess,
            t_done - t0,
        )
        return out

    @staticmethod
    def _cfgrib_missing_var_reason(ds: xr.Dataset, var_name: str) -> bool:
        if "_missing_var_name" not in ds.attrs:
            return False
        return ds.attrs["_missing_var_name"] == var_name and "_has_var" in ds

    @staticmethod
    def _cfgrib_has_requested_var(ds: xr.Dataset) -> bool:
        if "_has_var" not in ds:
            return False
        return bool(ds["_has_var"].any().item())

    def _candidate_files_for_date(
        self,
        date,
        *,
        include_exact: bool = True,
    ) -> list[tuple[str, list[Path]]]:
        attempts = self._candidate_file_attempts_for_date(date, include_exact=include_exact)
        return [(attempt["kind"], attempt["files"]) for attempt in attempts if attempt["files"]]

    def _candidate_file_attempts_for_date(
        self,
        date,
        *,
        include_exact: bool = True,
    ) -> list[dict[str, object]]:
        previous_date = pd.Timestamp(date) - self.leadtime.to_timedelta()
        root_path, date_config = self._resolve_path_config(previous_date)
        data_path = self._build_data_path(
            config=date_config,
            root_path=root_path,
            date=previous_date,
        )
        realizations = date_config.realizations
        attempts: list[dict[str, object]] = []

        if include_exact:
            exact_glob = self._build_data_glob(
                config=date_config,
                previous_date=previous_date,
                current_date=date,
            )
            exact_files = self._latest_matching_files(data_path, exact_glob, realizations)
            attempts.append({
                "kind": "exact",
                "target_date": pd.Timestamp(date),
                "data_path": data_path,
                "glob": exact_glob,
                "files": exact_files,
            })

        allow_shifted_fallback = not self.per_product_regrid

        if allow_shifted_fallback and date_config.minus_timedelta is not None:
            minus_date = pd.Timestamp(date) - date_config.minus_timedelta
            minus_glob = self._build_data_glob(
                config=date_config,
                previous_date=previous_date,
                current_date=minus_date,
            )
            minus_files = self._latest_matching_files(data_path, minus_glob, realizations)
            attempts.append({
                "kind": "minus",
                "target_date": minus_date,
                "data_path": data_path,
                "glob": minus_glob,
                "files": minus_files,
            })

        if allow_shifted_fallback and date_config.plus_timedelta is not None:
            plus_date = pd.Timestamp(date) + date_config.plus_timedelta
            plus_glob = self._build_data_glob(
                config=date_config,
                previous_date=previous_date,
                current_date=plus_date,
            )
            plus_files = self._latest_matching_files(data_path, plus_glob, realizations)
            attempts.append({
                "kind": "plus",
                "target_date": plus_date,
                "data_path": data_path,
                "glob": plus_glob,
                "files": plus_files,
            })

        return attempts

    def _open_one_cfgrib_with_var_fallback(self, path: Path, var_name: str, date) -> xr.Dataset:
        ds = self._open_one_cfgrib(path, var_name, date)
        if self._cfgrib_has_requested_var(ds) or not self._cfgrib_missing_var_reason(ds, var_name):
            return ds

        candidate_files = self._candidate_files_for_date(date, include_exact=False)
        checked_paths: list[str] = []

        for source_kind, candidate_group in candidate_files:
            for candidate in candidate_group:
                if candidate == path:
                    continue
                checked_paths.append(candidate.name)
                ds_candidate = self._open_one_cfgrib(candidate, var_name, date)
                if self._cfgrib_has_requested_var(ds_candidate):
                    if source_kind == "minus":
                        self.elements.extra.setdefault("minus_samples", []).append(date)
                    elif source_kind == "plus":
                        self.elements.extra.setdefault("plus_samples", []).append(date)
                    logger.info(
                        "%s Recovered missing var=%s for date=%s using %s fallback file %s instead of %s",
                        SOURCE_CTX,
                        var_name,
                        date,
                        source_kind,
                        candidate,
                        path,
                    )
                    return ds_candidate

        logger.warning(
            "%s Requested var=%s missing in %s for date=%s; no valid %s fallback found",
            SOURCE_CTX,
            var_name,
            path,
            date,
            checked_paths,
        )
        return ds

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

    def _product_key_for_date(self, date) -> tuple[str, str, str, str, str, str, str]:
        previous_date = pd.Timestamp(date) - self.leadtime.to_timedelta()
        root_path, file_name_config = self._resolve_path_config(previous_date)
        return (
            str(Path(root_path)),
            file_name_config.file_header,
            file_name_config.file_suffix,
            file_name_config.file_path_header,
            file_name_config.file_path_suffix,
            file_name_config.file_date_format,
            file_name_config.date_separator,
        )

    @staticmethod
    def _product_label(product_key: tuple[str, str, str, str, str, str, str]) -> str:
        root_path, file_header, file_suffix, *_ = product_key
        root_name = Path(root_path).name or root_path
        return f"{root_name}:{file_header}{file_suffix}"

    def _group_dates_by_product(self, dates: list[pd.Timestamp]) -> dict[tuple[str, str, str, str, str, str, str], list[pd.Timestamp]]:
        groups: dict[tuple[str, str, str, str, str, str, str], list[pd.Timestamp]] = {}
        for date in dates:
            key = self._product_key_for_date(date)
            groups.setdefault(key, []).append(date)
        return groups

    def _open_sample_dataset(
        self,
        sample: list[Path],
        date,
        *,
        realization_concat_dim: str,
        selected_vars: list,
        has_shifted_samples: bool,
        lock,
        apply_regrid: bool,
    ) -> xr.Dataset | None:
        t0 = time.perf_counter()
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
                            return self._open_one_cfgrib_with_var_fallback(path, var_name, date)

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
                        return self._open_one_cfgrib_with_var_fallback(path, var_name, date)

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

        ds_sample = self._normalize_sample_dataset(ds_sample, date, apply_regrid=apply_regrid)
        logger.debug(
            "%s sample load timing date=%s files=%s total=%.3fs",
            SOURCE_CTX,
            date,
            len(sample),
            time.perf_counter() - t0,
        )
        return ds_sample

    def _normalize_sample_dataset(self, ds_sample: xr.Dataset, date, *, apply_regrid: bool) -> xr.Dataset | None:
        t0 = time.perf_counter()
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

        if apply_regrid and self.sample_regrid_resolution is not None:
            ds_sample = ds_sample.earthml.regrid_to_rectilinear(
                region=self.data_selection.region,
                resolution=self.sample_regrid_resolution,
                vars_to_regrid=self.regrid_vars,
                silent=True,
            )

        logger.debug(
            "%s sample normalize timing date=%s dims=%s total=%.3fs",
            SOURCE_CTX,
            date,
            dict(ds_sample.sizes),
            time.perf_counter() - t0,
        )
        return ds_sample

    def _get_data_filenames(
        self,
        config: JunoLocalSourceFileNameConfig,
    ) -> Sample:
        """Get the data filenames for the given data selection."""

        s = Sample(extra={"plus_samples": [], "minus_samples": []})
        assert config.realizations == "all" or config.realizations > 0

        for date in self.date_range:
            attempts = self._candidate_file_attempts_for_date(date, include_exact=True)
            logger.debug("%s Sample [%s] candidate file names: %s", SOURCE_CTX, date, str(attempts))
            found = False

            for attempt in attempts:
                source_kind = attempt["kind"]
                files = attempt["files"]
                if not files:
                    continue
                s.samples[date] = files
                if source_kind == "minus":
                    s.extra["minus_samples"].append(date)
                elif source_kind == "plus":
                    s.extra["plus_samples"].append(date)
                found = True
                break

            if not found:
                logger.warning("%s Missed sample (local filename not found): %s", SOURCE_CTX, date)
                for attempt in attempts:
                    logger.debug(
                        "%s Missed sample details date=%s attempt=%s target_date=%s path=%s glob=%s files=%s",
                        SOURCE_CTX,
                        date,
                        attempt["kind"],
                        attempt["target_date"],
                        attempt["data_path"],
                        attempt["glob"],
                        [str(path.name) for path in attempt["files"]],
                    )
                s.missed.add(date)

        return s

    def _load_samples_for_dates(
        self,
        *,
        dates: list[pd.Timestamp],
        realization_concat_dim: str,
        selected_vars: list,
        has_shifted_samples: bool,
        lock,
        apply_regrid: bool,
        progress_label: str,
    ) -> dict[pd.Timestamp, xr.Dataset]:
        samples_d: dict[pd.Timestamp, xr.Dataset] = {}
        samples = [self.elements.samples[date] for date in dates]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=get_console(),
        ) as prog:
            task = prog.add_task(progress_label, total=len(samples))

            def _load_for_date(sample, date):
                assert isinstance(sample, list), f"Sample should be a list but it is {type(sample)}"
                return self._open_sample_dataset(
                    sample,
                    date,
                    realization_concat_dim=realization_concat_dim,
                    selected_vars=selected_vars,
                    has_shifted_samples=has_shifted_samples,
                    lock=lock,
                    apply_regrid=apply_regrid,
                )

            if self.file_open_workers == 1:
                for sample, date in zip(samples, dates):
                    ds_sample = _load_for_date(sample, date)
                    if ds_sample is None:
                        self.elements.missed.add(date)
                        logger.debug("%s Juno local sample [%s] returned None", SOURCE_CTX, date)
                    else:
                        samples_d[date] = ds_sample
                        logger.debug("%s Juno local, vars in sample [%s]: %s", SOURCE_CTX, date, ds_sample.data_vars)
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
                            logger.debug("%s Juno local sample [%s] returned None", SOURCE_CTX, date)
                        else:
                            samples_d[date] = ds_sample
                            logger.debug("%s Juno local, vars in sample [%s]: %s", SOURCE_CTX, date, ds_sample.data_vars)
                        prog.advance(task)

        return samples_d

    def _align_samples_to_shared_lon_grid(
        self,
        samples_d: dict[Any, xr.Dataset],
        *,
        label: str,
    ) -> dict[Any, xr.Dataset]:
        if not samples_d:
            return samples_d

        valid_keys = sorted(samples_d)
        ref_key = valid_keys[0]
        ref_lon_name = samples_d[ref_key].earthml.guessed_coords.longitude
        ref_lon_da = samples_d[ref_key][ref_lon_name]
        ref_lon = np.asarray(ref_lon_da.values, dtype=np.float64)
        ref_lon = np.round(ref_lon, 10)

        if ref_lon_da.ndim == 1:
            ref_lon = np.unique(ref_lon)
            logger.info("%s %s inferred reference 1D lon size: %s", SOURCE_CTX, label, ref_lon.size)
        elif ref_lon_da.ndim == 2:
            logger.info("%s %s inferred reference 2D lon shape: %s", SOURCE_CTX, label, ref_lon.shape)
        else:
            raise ValueError(
                f"{ref_key}: unsupported reference longitude coord {ref_lon_name!r} "
                f"with ndim={ref_lon_da.ndim}"
            )

        out: dict[Any, xr.Dataset] = {}
        for key, ds_sample in samples_d.items():
            lon_name = ds_sample.earthml.guessed_coords.longitude
            lon_da = ds_sample[lon_name]
            vals = np.asarray(lon_da.values, dtype=np.float64)
            vals = np.round(vals, 10)

            if lon_da.ndim != ref_lon_da.ndim:
                raise ValueError(
                    f"{key}: longitude ndim mismatch {lon_da.ndim} vs ref {ref_lon_da.ndim}"
                )

            if lon_da.ndim == 1:
                if vals.size != ref_lon.size:
                    raise ValueError(
                        f"{key}: longitude size mismatch {vals.size} vs ref {ref_lon.size}"
                    )
                if not np.allclose(vals, ref_lon, atol=1e-8, rtol=0.0):
                    diff = np.max(np.abs(vals - ref_lon))
                    raise ValueError(
                        f"{key}: longitude grid differs from reference, max abs diff={diff}"
                    )
                out[key] = ds_sample.assign_coords({lon_name: ref_lon})
                continue

            if vals.shape != ref_lon.shape:
                raise ValueError(
                    f"{key}: longitude shape mismatch {vals.shape} vs ref {ref_lon.shape}"
                )
            if not np.allclose(vals, ref_lon, atol=1e-8, rtol=0.0):
                diff = np.max(np.abs(vals - ref_lon))
                raise ValueError(
                    f"{key}: longitude grid differs from reference, max abs diff={diff}"
                )
            out[key] = ds_sample.assign_coords({lon_name: (lon_da.dims, ref_lon)})

        return out

    def _trim_samples_to_valid_realizations(
        self,
        samples_d: dict[pd.Timestamp, xr.Dataset],
        *,
        realization_concat_dim: str,
    ) -> dict[pd.Timestamp, xr.Dataset]:
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
        filtered: dict[pd.Timestamp, xr.Dataset] = {}

        for date, ds, has_any in zip(validity_dates, (samples_d[d] for d in validity_dates), validity_values):
            ok = bool(has_any.item())
            if ok:
                filtered[date] = ds
                samples_len.append(ds.sizes.get(realization_concat_dim, 1))
            else:
                missing_samples.append(date)

        if not samples_len:
            raise ValueError(
                f"No valid samples found for {self.source_name}. "
                f"Total candidates={len(samples_d)}, missing_samples={len(missing_samples)}"
            )

        min_R = min(samples_len)
        for date, ds in filtered.items():
            if realization_concat_dim in ds.dims:
                dsR = ds.isel({realization_concat_dim: slice(0, min_R)})
                R = dsR.sizes[realization_concat_dim]
                filtered[date] = dsR.assign_coords({realization_concat_dim: np.arange(R)})

        self.elements.missed.update(pd.to_datetime(missing_samples).to_pydatetime().tolist())
        return filtered

    def _concat_samples_by_time(
        self,
        samples_d: dict[pd.Timestamp, xr.Dataset],
    ) -> xr.Dataset:
        if not samples_d:
            raise ValueError(
                f"No datasets left to concatenate for {self.source_name} after removing missed samples."
            )

        ref_date = sorted(samples_d)[0]
        concat_time_dim = (
            samples_d[ref_date].earthml.guessed_dims.time
            or samples_d[ref_date].earthml.guessed_coords.time
            or "time"
        )
        times = np.array(sorted(samples_d.keys()), dtype="datetime64[ns]")
        objs = [samples_d[d] for d in sorted(samples_d)]
        ds_all = xr.concat(
            objs=objs,
            dim=xr.IndexVariable(concat_time_dim, times),
            coords="minimal",
            compat="override",
            join="outer",
            combine_attrs="drop_conflicts",
        )
        return ds_all


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
        if self.per_product_regrid and self.sample_regrid_resolution is not None:
            product_groups = self._group_dates_by_product(dates)
            logger.info(
                "%s Per-product regrid enabled: %s active product groups",
                SOURCE_CTX,
                len(product_groups),
            )
            for product_key, group_dates in product_groups.items():
                logger.info(
                    "%s Product group %s has %s samples",
                    SOURCE_CTX,
                    self._product_label(product_key),
                    len(group_dates),
                )

            product_datasets: dict[pd.Timestamp, xr.Dataset] = {}
            for product_key, group_dates in product_groups.items():
                label = self._product_label(product_key)
                group_samples_d = self._load_samples_for_dates(
                    dates=group_dates,
                    realization_concat_dim=realization_concat_dim,
                    selected_vars=selected_vars,
                    has_shifted_samples=has_shifted_samples,
                    lock=lock,
                    apply_regrid=False,
                    progress_label=f"Opening local Juno samples ({label})",
                )
                if not group_samples_d:
                    logger.warning("%s Product group %s produced no valid native samples", SOURCE_CTX, label)
                    continue

                dbg_sample_k = next(iter(group_samples_d))
                logger.debug(
                    "%s Juno local, size of native ds[%s] for %s after open_mfdataset: %s",
                    SOURCE_CTX,
                    dbg_sample_k,
                    label,
                    group_samples_d[dbg_sample_k].sizes,
                )

                group_samples_d = self._align_samples_to_shared_lon_grid(group_samples_d, label=f"{label} native")
                group_samples_d = self._trim_samples_to_valid_realizations(
                    group_samples_d,
                    realization_concat_dim=realization_concat_dim,
                )
                if not group_samples_d:
                    logger.warning("%s Product group %s left no native samples after validity filtering", SOURCE_CTX, label)
                    continue

                product_ds = self._concat_samples_by_time(group_samples_d)
                logger.info("%s Regridding product group %s after native concat", SOURCE_CTX, label)
                product_ds = product_ds.earthml.regrid_to_rectilinear(
                    region=self.data_selection.region,
                    resolution=self.sample_regrid_resolution,
                    vars_to_regrid=self.regrid_vars,
                    silent=True,
                )
                product_datasets[min(group_samples_d)] = product_ds

            if not product_datasets:
                raise ValueError(f"No product datasets left to concatenate for {self.source_name}.")

            product_datasets = self._align_samples_to_shared_lon_grid(
                product_datasets,
                label="regridded product datasets",
            )

            global_min_R = min(ds.sizes.get(realization_concat_dim, 1) for ds in product_datasets.values())
            for key, ds in product_datasets.items():
                if realization_concat_dim in ds.dims:
                    dsR = ds.isel({realization_concat_dim: slice(0, global_min_R)})
                    R = dsR.sizes[realization_concat_dim]
                    product_datasets[key] = dsR.assign_coords({realization_concat_dim: np.arange(R)})

            first_key = sorted(product_datasets)[0]
            concat_time_dim = (
                product_datasets[first_key].earthml.guessed_dims.time
                or product_datasets[first_key].earthml.guessed_coords.time
                or "time"
            )
            ds_all = xr.concat(
                objs=[product_datasets[k] for k in sorted(product_datasets)],
                dim=concat_time_dim,
                coords="minimal",
                compat="override",
                join="outer",
                combine_attrs="drop_conflicts",
            )
            logger.info("%s Juno local, size of ds after per-product concat+regrid: %s", SOURCE_CTX, ds_all.sizes)
            return ds_all

        samples_d = self._load_samples_for_dates(
            dates=dates,
            realization_concat_dim=realization_concat_dim,
            selected_vars=selected_vars,
            has_shifted_samples=has_shifted_samples,
            lock=lock,
            apply_regrid=self.per_sample_regrid,
            progress_label="Opening local Juno samples",
        )

        if samples_d:
            dbg_sample_k = next(iter(samples_d))
            logger.debug(
                "%s Juno local, size of ds[%s] after open_mfdataset: %s",
                SOURCE_CTX,
                dbg_sample_k,
                samples_d[dbg_sample_k].sizes,
            )

        samples_d = self._align_samples_to_shared_lon_grid(samples_d, label="Juno local")
        samples_d = self._trim_samples_to_valid_realizations(
            samples_d,
            realization_concat_dim=realization_concat_dim,
        )
        ds_all = self._concat_samples_by_time(samples_d)
        logger.info("%s Juno local, size of ds after concat: %s", SOURCE_CTX, ds_all.sizes)

        return ds_all
