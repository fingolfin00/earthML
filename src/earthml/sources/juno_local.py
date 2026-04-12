from typing import Literal
from dataclasses import dataclass
from functools import partial

import re, time, dask
from heapq import nlargest
from pathlib import Path

import hashlib
import json

from datetime import timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr

from dask.distributed import progress
from dask.utils import SerializableLock

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

from .dataclasses import DataSource, Sample, RegridConfig
from .utils import retry_fetch_after_hdf_err
from .xarray_local import MFXarrayLocalSource
from ._preprocess import preprocess_mfdataset
from ..logging import get_logger


REALIZATION_RE = re.compile(r"_r(\d+)")
logger = get_logger(__name__)


@dataclass
class JunoLocalSourceFileNameConfig:
    file_path_date_format               : str
    file_header                         : str
    file_suffix                         : str
    file_date_format                    : str
    both_data_and_previous_date_in_file : bool = True
    date_order                          : Literal["previous_current", "current_previous"] = "previous_current"
    date_separator                      : str = ""
    realizations                        : int | Literal["all"] = 1
    minus_timedelta                     : timedelta | None = None
    plus_timedelta                      : timedelta | None = None

@dataclass
class JunoLocalSourceConfig:
    leadtime                    : relativedelta # TODO move to Leadtime?
    root_path                   : str | Path
    engine                      : str
    file_name_config            : JunoLocalSourceFileNameConfig
    regrid_config               : RegridConfig
    select_area_after_request   : bool = True
    cfgrib_idx_path             : str | Path = ""


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

        self.regrid_resolution = config.regrid_config.regrid_resolution
        self.regrid_vars = config.regrid_config.regrid_vars if config.regrid_config.regrid_vars is not None else self.var_name_list

        self.cfgrib_idx_path = str(config.cfgrib_idx_path)
        if self.engine == "cfgrib" and self.cfgrib_idx_path:
            Path(self.cfgrib_idx_path).mkdir(parents=True, exist_ok=True)


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


    def _open_one_cfgrib(self, path: Path, var_name: str, date, lock):
        filter_by_keys = {"cfVarName": var_name}
        indexpath_key = self._make_cfgrib_index_key(path, filter_by_keys)
        indexpath = str(Path(self.cfgrib_idx_path) / f"{indexpath_key}.idx")

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

        # Keep exact same observable behavior: newest files only, newest-first order
        top = nlargest(realizations, matches, key=lambda t: t[0])
        top.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in top]

    @staticmethod
    def _realization_key(p: Path) -> int:
        m = REALIZATION_RE.search(p.name)
        if not m:
            raise ValueError(f"Could not parse realization from filename: {p.name}")
        return int(m.group(1))


    def _get_data_filenames(
        self,
        leadtime: relativedelta,
        config: JunoLocalSourceFileNameConfig,
    ) -> Sample:
        """Get the data filenames for the given data selection."""

        s = Sample(extra={"plus_samples": [], "minus_samples": []})
        assert config.realizations == "all" or config.realizations > 0

        both_dates_in_name = config.both_data_and_previous_date_in_file
        date_order = config.date_order
        date_separator = config.date_separator
        realizations = config.realizations
        file_header = config.file_header
        file_suffix = config.file_suffix
        file_date_format = config.file_date_format
        file_path_date_format = config.file_path_date_format

        minus_td = config.minus_timedelta
        plus_td = config.plus_timedelta

        for date in self.date_range:
            previous_date = date - leadtime
            data_path = self.path.joinpath(previous_date.strftime(file_path_date_format))

            prev_str = previous_date.strftime(file_date_format)
            date_str = date.strftime(file_date_format)

            if both_dates_in_name:
                if date_order == "current_previous":
                    data_glob = f"{file_header}{date_str}{date_separator}{prev_str}{file_suffix}"
                else:
                    data_glob = f"{file_header}{prev_str}{date_separator}{date_str}{file_suffix}"
            else:
                data_glob = f"{file_header}{prev_str}{file_suffix}"

            files_exact = self._latest_matching_files(data_path, data_glob, realizations)

            if files_exact:
                s.samples[date] = files_exact
                continue

            found = False

            if minus_td is not None:
                test_date = date - minus_td
                if both_dates_in_name:
                    test_date_str = test_date.strftime(file_date_format)
                    if date_order == "current_previous":
                        test_glob = f"{file_header}{test_date_str}{date_separator}{prev_str}{file_suffix}"
                    else:
                        test_glob = f"{file_header}{prev_str}{date_separator}{test_date_str}{file_suffix}"
                else:
                    test_glob = f"{file_header}{prev_str}{file_suffix}"

                test_files = self._latest_matching_files(data_path, test_glob, realizations)
                if test_files:
                    s.samples[date] = test_files
                    s.extra["minus_samples"].append(date)
                    found = True

            if not found and plus_td is not None:
                test_date = date + plus_td
                if both_dates_in_name:
                    test_date_str = test_date.strftime(file_date_format)
                    if date_order == "current_previous":
                        test_glob = f"{file_header}{test_date_str}{date_separator}{prev_str}{file_suffix}"
                    else:
                        test_glob = f"{file_header}{prev_str}{date_separator}{test_date_str}{file_suffix}"
                else:
                    test_glob = f"{file_header}{prev_str}{file_suffix}"

                test_files = self._latest_matching_files(data_path, test_glob, realizations)
                if test_files:
                    s.samples[date] = test_files
                    s.extra["plus_samples"].append(date)
                    found = True

            if not found:
                logger.warning("Missed sample (local filename not found): %s", date)
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
            "Juno local file samples: %s, minus: %s, plus: %s, missed: %s",
            len(samples),
            len(minus_samples),
            len(plus_samples),
            len(self.elements.missed),
        )

        samples_d: dict[pd.Timestamp, xr.Dataset] = {}
        lock = SerializableLock()

        common_args = {
            "combine": "nested",
            "coords": "minimal" if has_shifted_samples else "different",
            # if minus/plus_samples time coordinate stepping might be irregular so override
            "compat": "override" if has_shifted_samples else "no_conflicts",
            "engine": self.engine,
            # "chunks": {'time': -1}, ## speed-up attempt
            "chunks": "auto",
            "parallel": True,
            # "parallel": False, ## speed-up attempt
            "decode_timedelta": True,
            "backend_kwargs": {},
            "preprocess": partial(preprocess_mfdataset, data=self.data_selection),
            "decode_cf": True,
            "errors": "warn",
            "lock": lock,
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
        ) as prog:
            task = prog.add_task("Opening local Juno samples", total=len(samples))

            for sample, date in zip(samples, dates):
                assert isinstance(sample, list), f"Sample should be a list but it is {type(sample)}"

                # Try sorting by realization "...rxx..."
                if len(sample) > 1:
                    sample = sorted(sample, key=self._realization_key)

                args = dict(common_args) # make local copy
                args["paths"] = sample

                if is_var_list:
                    var_ds_list = []

                    for var in selected_vars:
                        # var_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": var.name}} # not currently possible to filter with a list of keys (see https://github.com/ecmwf/cfgrib/issues/138)
                        # Use open_dataset for grib
                        if self.engine == "cfgrib":
                            per_file = []
                            for path in sample:
                                def _open_one(path=path, var_name=var.name, date=date):
                                    return self._open_one_cfgrib(path, var_name, date, lock)

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

                        else: # use open_mfdataset if not grib
                            var_args = dict(args) # again local copy to avoid leakage between iterations
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

                    # Use open_dataset for grib
                    if self.engine == "cfgrib":
                        per_file = []
                        for path in sample:
                            def _open_one(path=path, var_name=var.name, date=date):
                                return self._open_one_cfgrib(path, var_name, date, lock)

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

                    else: # use open_mfdataset if not grib
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

                    # Load realization coord
                    if realization_concat_dim in ds_sample.coords:
                        ds_sample = ds_sample.assign_coords(
                            {realization_concat_dim: np.asarray(ds_sample[realization_concat_dim].values)}
                        )
                    # Promote realization to coord
                    if realization_concat_dim in ds_sample.dims:
                        ds_sample = ds_sample.assign_coords(
                            {realization_concat_dim: ds_sample[realization_concat_dim]}
                        )

                # Reject placeholder / invalid samples before touching spatial coords
                if "_has_var" not in ds_sample:
                    raise ValueError(f"{date}: ds_sample has no _has_var flag")

                if not bool(ds_sample["_has_var"].any().item()):
                    self.elements.missed.add(date)
                    prog.advance(task)
                    continue

                lon_name = ds_sample.earthml.guessed_coords.longitude
                lat_name = ds_sample.earthml.guessed_coords.latitude

                if lon_name is None or lat_name is None:
                    self.elements.missed.add(date)
                    prog.advance(task)
                    continue

                lon = np.asarray(ds_sample[lon_name].values, dtype=np.float64)

                # normalize convention
                lon = ((lon + 180.0) % 360.0) - 180.0
                lon = np.round(lon, 10)
                lon = np.where(np.isclose(lon, 180.0, atol=1e-8), -180.0, lon)

                ds_sample = ds_sample.assign_coords({lon_name: lon}).sortby(lon_name)

                vals = np.asarray(ds_sample[lon_name].values)
                _, idx = np.unique(vals, return_index=True)
                ds_sample = ds_sample.isel({lon_name: np.sort(idx)})

                samples_d[date] = ds_sample
                prog.advance(task)

        # dbg_sample_k = self.date_range[10]
        # print(f"Juno local, size of ds[{dbg_sample_k}] after open_mfdataset: {samples_d[dbg_sample_k].sizes}")

        # Infer resolution from first valid sample
        valid_dates = sorted(samples_d)
        ref_date = valid_dates[0]
        ref_lon_name = samples_d[ref_date].earthml.guessed_coords.longitude
        ref_lon = np.asarray(samples_d[ref_date][ref_lon_name].values, dtype=np.float64)
        ref_lon = np.round(ref_lon, 10)
        ref_lon = np.unique(ref_lon) # remove exact duplicates

        logger.info("Juno local inferred reference lon size: %s", ref_lon.size)

        # Enforce same lon convention to all samples
        for date, ds_sample in samples_d.items():
            lon_name = ds_sample.earthml.guessed_coords.longitude
            vals = np.asarray(ds_sample[lon_name].values, dtype=np.float64)
            vals = np.round(vals, 10)

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
        logger.info("Juno local, size of ds after concat: %s", ds_all.sizes)

        return ds_all
