from typing import Literal
from dataclasses import dataclass
from functools import partial

import re, time, dask
from heapq import nlargest
from pathlib import Path

from datetime import timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr

from dask.utils import SerializableLock

from rich import print
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


REALIZATION_RE = re.compile(r"_r(\d+)")


@dataclass
class JunoLocalSourceFileNameConfig:
    file_path_date_format               : str
    file_header                         : str
    file_suffix                         : str
    file_date_format                    : str
    both_data_and_previous_date_in_file : bool = True
    realizations                        : int | Literal["all"] = 1
    minus_timedelta                     : timedelta | None = None
    plus_timedelta                      : timedelta | None = None

@dataclass
class JunoLocalSourceConfig:
    leadtime            : relativedelta # TODO move to Leadtime?
    root_path           : str | Path
    engine              : str
    file_name_config    : JunoLocalSourceFileNameConfig
    regrid_config       : RegridConfig


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

        self.regrid_resolution = config.regrid_config.regrid_resolution
        self.regrid_vars = config.regrid_config.regrid_vars if config.regrid_config.regrid_vars is not None else self.var_name_list


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
                data_glob = f"{file_header}{prev_str}{date_str}{file_suffix}"
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
                    test_glob = f"{file_header}{prev_str}{test_date.strftime(file_date_format)}{file_suffix}"
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
                    test_glob = f"{file_header}{prev_str}{test_date.strftime(file_date_format)}{file_suffix}"
                else:
                    test_glob = f"{file_header}{prev_str}{file_suffix}"

                test_files = self._latest_matching_files(data_path, test_glob, realizations)
                if test_files:
                    s.samples[date] = test_files
                    s.extra["plus_samples"].append(date)
                    found = True

            if not found:
                print(f"Missed sample (local filename not found): {date}")
                s.missed.add(date)

        return s


    def _get_data(self) -> xr.Dataset:
        samples = [s for date, s in self.elements.samples.items() if date not in self.elements.missed]
        assert len(samples) > 0, "No samples obtained."
        dates = [date for date in self.elements.samples.keys() if date not in self.elements.missed]

        minus_samples = self.elements.extra["minus_samples"]
        plus_samples = self.elements.extra["plus_samples"]
        has_shifted_samples = bool(minus_samples or plus_samples)

        print(
            f"Juno local file samples: {len(samples)}, "
            f"minus: {len(minus_samples)}, plus: {len(plus_samples)}, missed: {len(self.elements.missed)}"
        )

        samples_d: dict[pd.Timestamp, xr.Dataset] = {}
        lock = SerializableLock()

        common_args = {
            "combine": "nested",
            "coords": "minimal" if has_shifted_samples else "different",
            # if minus/plus_samples time coordinate stepping might be irregular so override
            "compat": "override" if has_shifted_samples else "no_conflicts",
            "engine": self.engine,
            # "chunks": {'time': -1},
            "chunks": "auto",
            "parallel": True,
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
            task = prog.add_task("Opening local samples", total=len(samples))

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
                        var_args = dict(args) # again local copy to avoid leakage between iterations

                        if self.engine == "cfgrib":
                            var_args["backend_kwargs"] = {
                                "indexpath": "", # disable .idx writing
                                "filter_by_keys": {"cfVarName": var.name},
                            }

                        var_args["preprocess"] = partial(
                            preprocess_mfdataset,
                            data=self.data_selection,
                            var_name=var.name,
                            date=date,
                        )
                        # var_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": var.name}} # not currently possible to filter with a list of keys (see https://github.com/ecmwf/cfgrib/issues/138)
                        # var_args["indexpath"] = ""

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
                            {realization_concat_dim: ds_sample[realization_concat_dim].load()}
                        )
                else:
                    var = selected_vars[0]

                    if self.engine == "cfgrib":
                        args["backend_kwargs"] = {
                            "indexpath": "", # disable .idx writing
                            "filter_by_keys": {"cfVarName": var.name},
                        }

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
                            {realization_concat_dim: ds_sample[realization_concat_dim].load()}
                        )
                    # Promote realization to coord
                    if realization_concat_dim in ds_sample.dims:
                        ds_sample = ds_sample.assign_coords(
                            {realization_concat_dim: ds_sample[realization_concat_dim]}
                        )

                samples_d[date] = ds_sample
                prog.advance(task)

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

        # Subset (select region)
        ds = ds.earthml.subset(self.data_selection)

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

        # TODO add progress bar to concat too?
        return xr.concat(
            objs=objs,
            dim=xr.IndexVariable(time_dim, times), # TODO time_dim here is an abuse, as is selected in a loop
            coords="minimal",
            compat="override",
            join="outer",
            combine_attrs="drop_conflicts",
        )
