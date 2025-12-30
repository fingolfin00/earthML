from datetime import timedelta
from dateutil.relativedelta import relativedelta
from functools import partial
import re, time
import numpy as np
import pandas as pd
import xarray as xr
from dask.utils import SerializableLock
from rich import print

from ..dataclasses import DataSource, Sample
from ..utils import retry_fetch_after_hdf_err, get_ds_resolution, subset_ds, regrid_to_rectilinear, _guess_coord_name
from .xarray_local import MFXarrayLocalSource
from ._preprocess import preprocess_mfdataset

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
        assert len(samples) > 0, "No samples obtained."
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
            "preprocess": partial(preprocess_mfdataset, data=self.data_selection),
            "decode_cf": True,
            "errors": "warn",
            "lock": lock,
        }
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
                    common_args["preprocess"] = partial(preprocess_mfdataset, data=self.data_selection, var_name=var.name, date=date)

                    def _open_mfdataset ():
                        return xr.open_mfdataset(**common_args)
                    var_ds_list.append(retry_fetch_after_hdf_err(_open_mfdataset, error_re=r"Unspecified error in H5DSget_num_scales.*"))

                ds_sample = xr.merge(var_ds_list, compat="no_conflicts", combine_attrs="no_conflicts")

                # Load time coord
                if self.concat_dim in ds_sample.coords:
                    ds_sample = ds_sample.assign_coords({self.concat_dim: ds_sample[self.concat_dim].load()})
            else:
                if self.engine == "cfgrib":
                    common_args["backend_kwargs"] = {"filter_by_keys": {"cfVarName": self.data_selection.variable.name}}
                    common_args["indexpath"] = ""

                common_args["preprocess"] = partial(preprocess_mfdataset, data=self.data_selection, var_name=self.data_selection.variable.name, date=date)

                # Tested support for netcdf4
                def _open_mfdataset ():
                    return xr.open_mfdataset(**common_args)

                ds_sample = retry_fetch_after_hdf_err(_open_mfdataset, error_re=r"Unspecified error in H5DSget_num_scales.*")

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

        # Count extra missed in preprocess
        missing_mask, missing_times = [], []
        for d in list(sorted(samples_d)):
            ds = samples_d[d]
            missing_mask.append(ds["_has_var"].values)
            if ~ds["_has_var"]:
                time_coord = _guess_coord_name(ds, "time", ["valid_time", "time_counter"])
                if time_coord is None and "source_time" in ds.data_vars:
                    missing_times.append(ds["source_time"].values)
                else:
                    raise ValueError("Couldn't find time coord in processed dataset, aborting.")

        # Flatten arrays
        missing_mask = np.array([a.item() for a in missing_mask])
        missing_times = [np.asarray(x).item() for x in missing_times]

        # Number of missing timesteps
        n_missing = int(sum(~missing_mask))
        if n_missing > 0:
            print(f"Missed {n_missing} timesteps during preprocess")
            # List the times that were missing
            # missing_times = ds["time"].where(missing_mask, drop=True).values
            # print(missing_times)
            self.elements.missed.update(pd.to_datetime(missing_times).to_pydatetime().tolist())
            # # Keep only valid timesteps
            # ds = ds.where(ds["_has_var"], drop=True)
        # print(self.elements.missed)

        # Concatenate
        times = np.array([d for d in sorted(samples_d.keys()) if d not in self.elements.missed], dtype="datetime64[ns]")
        objs = [samples_d[d] for d in sorted(samples_d) if d not in self.elements.missed]
        # print(objs)
        ds = xr.concat(
            objs=objs,
            dim=xr.IndexVariable(self.concat_dim, times) if self.concat_dim in ('time', 'valid_time', 'time_counter') else self.concat_dim,
            coords='minimal',
            # compat="broadcast_equals",
            compat="override",
            join='outer',
            # join='exact',
            combine_attrs='drop_conflicts'
        )

        # Add missed info to dataset
        missed_sorted = sorted(self.elements.missed)
        missed_np = np.array(missed_sorted, dtype="datetime64[ns]")
        # print(missed_np)
        ds = ds.assign(missed_time=("missed_time", missed_np))
        ds = ds.set_coords("missed_time")
        ds["missed_time"].encoding.update({
            "units": "nanoseconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        })
        # if "missed_time" in ds:
        #     print("just saved dtype:", ds["missed_time"].dtype)
        #     print("just saved head:", ds["missed_time"].values)

        ds = subset_ds(self.data_selection, ds)

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
