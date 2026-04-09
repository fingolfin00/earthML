from typing import Literal
from dataclasses import dataclass
from functools import partial

import re, time
from pathlib import Path

from datetime import timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr

from dask.utils import SerializableLock

from rich import print

from .dataclasses import DataSource, Sample, RegridConfig
from .utils import retry_fetch_after_hdf_err
from .xarray_local import MFXarrayLocalSource
from ._preprocess import preprocess_mfdataset


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

        self.select_area_after_request = self.config.select_area_after_request
        self.convert_unit = self.config.convert_unit

        self.regrid_resolution = config.regrid_config.regrid_resolution
        self.regrid_vars = config.regrid_config.regrid_vars if config.regrid_config.regrid_vars is not None else self.var_name_list


    def _get_data_filenames(
        self,
        leadtime: relativedelta,
        config: JunoLocalSourceFileNameConfig,
    ) -> Sample:
        """Get the data filenames for the given data selection."""

        s = Sample(extra={"plus_samples": [], "minus_samples": []})
        assert config.realizations == 'all' or config.realizations > 0
        for date in self.date_range:
            previous_date = date - leadtime
            # print("juno-local prev date:", previous_date)
            data_path = self.path.joinpath(previous_date.strftime(config.file_path_date_format))
            if config.both_data_and_previous_date_in_file:
                data_glob = f"{config.file_header}{previous_date.strftime(config.file_date_format)}{date.strftime(config.file_date_format)}{config.file_suffix}"
            else:
                data_glob = f"{config.file_header}{previous_date.strftime(config.file_date_format)}{config.file_suffix}"
            # print(f"{date}, glob:", data_path / data_glob)

            # files_exact = [p for p in data_path.glob(data_glob) if p.is_file()]
            files_exact = sorted(
                (p for p in data_path.glob(data_glob) if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True  # newest first
            )

            if len(files_exact) > 1:
                if config.realizations == 'all':
                    r = len(files_exact)
                else:
                    r = config.realizations
                # print(f"{date}: {len(files_exact)} matches")
                # print(f"{date}: {len(files_exact)} matches, keeping {files_exact[:r]}")
                files_exact = files_exact[:r]  # keep latest only, still a list
            # print(f"{date}, path:", files_exact)

            # Try direct match first
            if files_exact:
                s.samples[date] = files_exact
                continue

            # Fallback search
            found = False
            if config.minus_timedelta and config.plus_timedelta:
                # print(minus_timedelta, plus_timedelta)
                for delta, store in [
                    (-config.minus_timedelta, s.extra['minus_samples']),
                    (config.plus_timedelta, s.extra['plus_samples']),
                ]:
                    test_date = date + delta
                    if config.both_data_and_previous_date_in_file:
                        test_glob = f"{config.file_header}{previous_date.strftime(config.file_date_format)}{test_date.strftime(config.file_date_format)}{config.file_suffix}"
                    else:
                        test_glob = f"{config.file_header}{previous_date.strftime(config.file_date_format)}{config.file_suffix}" # TODO look better into this
                    test_files = [p for p in data_path.glob(test_glob) if p.is_file()]
                    # print(f"New file {delta}: {test_files}")
                    if test_files:
                        if config.realizations == 'all':
                            r = len(test_files)
                        else:
                            r = config.realizations
                        s.samples[date] = test_files[:r]
                        # s.samples.extend(test_files)
                        store.append(date)
                        found = True
                        break

            if not found:
                print(f"Missed sample (local filename not found): {date}")
                s.missed.add(date)

        return s

    def _get_data(self) -> xr.Dataset:
        # years = [str(date.year) for date in xr.date_range(start=data.period.start, end=data.period.end, freq='YS', inclusive='left')]
        # print(f"{self.source_name} missed dates: {self.elements.missed}")
        # print(self.elements.samples)
        samples = [s for date, s in self.elements.samples.items() if date not in self.elements.missed] # list of lists
        assert len(samples) > 0, "No samples obtained."
        dates = [date for date in self.elements.samples.keys() if date not in self.elements.missed]
        print(f"Juno local file samples: {len(samples)}, minus: {len(self.elements.extra['minus_samples'])}, plus: {len(self.elements.extra['plus_samples'])}, missed: {len(self.elements.missed)}")
        samples_d = {}

        lock = SerializableLock()

        common_args = {
            "combine": "nested",  # if self.concat_dim else "by_coords",
            "coords": "minimal" if (self.elements.extra['minus_samples'] or self.elements.extra['plus_samples']) else "different", # ["time"],
            # if minus/plus_samples time coordinate stepping might be irregular so override
            "compat": "override" if (self.elements.extra['minus_samples'] or self.elements.extra['plus_samples']) else "no_conflicts",
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

        probe_backend_kwargs = dict(common_args.get("backend_kwargs", {}))

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
            common_args["concat_dim"] = ds0.earthml.guessed_dims.realization or "realization"

        realization_concat_dim = common_args["concat_dim"]

        for sample, date in zip(samples, dates):
            assert isinstance(sample, list), f"Sample should be a list but it is {type(sample)}"

            # Try sorting by realization "...rxx..."
            if len(sample) > 1:
                sample = sorted(sample, key=lambda p: int(re.search(r"_r(\d+)", p.name).group(1)))
            # print(sample)
            common_args['paths'] = sample

            if isinstance(self.data_selection.variable, list):
                var_ds_list = []
                for var in self.data_selection.variable:
                    if self.engine == "cfgrib":
                        common_args["backend_kwargs"] = {
                            "indexpath": "",                 # disable .idx writing
                            "filter_by_keys": {"cfVarName": var.name},
                        }
                        # common_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": var.name}} # not currently possible to filter with a list of keys (see https://github.com/ecmwf/cfgrib/issues/138)
                        # common_args["indexpath"] = ""
                    common_args["preprocess"] = partial(preprocess_mfdataset, data=self.data_selection, var_name=var.name, date=date)

                    def _open_mfdataset ():
                        return xr.open_mfdataset(**common_args)
                    var_ds_list.append(retry_fetch_after_hdf_err(_open_mfdataset, error_re=r"Unspecified error in H5DSget_num_scales.*"))

                ds_sample = xr.merge(var_ds_list, compat="no_conflicts", combine_attrs="no_conflicts")

                # Load realization coord
                if realization_concat_dim in ds_sample.coords:
                    ds_sample = ds_sample.assign_coords({realization_concat_dim: ds_sample[realization_concat_dim].load()})
            else:
                if self.engine == "cfgrib":
                    common_args["backend_kwargs"] = {
                        "indexpath": "",                 # disable .idx writing
                        "filter_by_keys": {"cfVarName": self.data_selection.variable.name},
                    }
                    # common_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": self.data_selection.variable.name}}
                    # common_args["indexpath"] = ""

                common_args["preprocess"] = partial(preprocess_mfdataset, data=self.data_selection, var_name=self.data_selection.variable.name, date=date)

                # Tested support for netcdf4
                def _open_mfdataset ():
                    return xr.open_mfdataset(**common_args)

                ds_sample = retry_fetch_after_hdf_err(_open_mfdataset, error_re=r"Unspecified error in H5DSget_num_scales.*")

                # Load realization coord
                if realization_concat_dim in ds_sample.coords:
                    ds_sample = ds_sample.assign_coords({realization_concat_dim: ds_sample[realization_concat_dim].load()})
                # Promote realization to coord
                if realization_concat_dim in ds_sample.dims:
                    ds_sample = ds_sample.assign_coords({realization_concat_dim: ds_sample[realization_concat_dim]})

            # print(ds_sample)
            # ds_sample is alway Rx1xYxX
            samples_d[date] = ds_sample

        # Count valid realizations
        samples_len, missing_samples = [], []
        for date, ds in samples_d.items():
            time_dim = ds.earthml.guessed_dims.time
            # print(f"Realization dim: {realization_concat_dim}, time dim: {time_dim}")
            dims = tuple(
                d for d in (realization_concat_dim, time_dim)
                if d is not None and d in ds["_has_var"].dims
            )
            # print(f"Sample {date} _has_var dims: {dims}")
            if ds["_has_var"].any(dim=dims):
                samples_len.append(ds.sizes.get(realization_concat_dim, 1))
            else:
                # Store dates with no valid realizations
                missing_samples.append(date)

        # print(f"Samples length for realization minimization: {len(samples_len)}")
        min_R = min(samples_len)
        # Use only min_R realizations per sample
        for date, ds in samples_d.items():
            if realization_concat_dim in ds.dims:
                dsR = ds.isel(realization=slice(0, min_R))
                # Realizations are simple integers
                R = dsR.sizes[realization_concat_dim]
                samples_d[date] = dsR.assign_coords(realization=np.arange(R))
        # print(f"Smallest number of realizations for valid samples: {min_R}")
        # print(f"Missed samples after file-processing: {missing_samples}")
        # print(f"Missed elements before file-processing: {self.elements.missed}")
        self.elements.missed.update(pd.to_datetime(missing_samples).to_pydatetime().tolist())
        # print(f"Updated missed elements: {self.elements.missed}")

        # Concatenate
        times = np.array([d for d in sorted(samples_d.keys()) if d not in self.elements.missed], dtype="datetime64[ns]")
        objs = [samples_d[d] for d in sorted(samples_d) if d not in self.elements.missed]
        # print(objs)

        return xr.concat(
            objs=objs,
            dim=xr.IndexVariable("time", times),
            coords='minimal',
            # compat="broadcast_equals",
            compat="override",
            join='outer',
            # join='exact',
            combine_attrs='drop_conflicts'
        )
