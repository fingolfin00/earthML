from typing import Sequence, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr

from dask.diagnostics.progress import ProgressBar

from ..base import (
    LeadtimeUnit,
    ClimPeriod,
    Settings,
    open_zarr,
    open_zarr_var,
    T_Xarray,
    subset_dataset,
    get_and_subset_datasets,
    save_zarr,
    safe_chunk_spec,
)


def calculate_climatology(
    da: xr.DataArray | xr.Dataset,
    time_dim: str = "time",
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    rolling_window: int | None = None,
    rolling_center: bool = True,
    rolling_min_periods: int = 1,
    check_group_counts: bool = True,
) -> T_Xarray:
    if clim_period in ("dayofyear_hour", "day_hour", "month_hour"):
        if clim_period == "dayofyear_hour":
            group_period = "dayofyear"
        elif clim_period == "day_hour":
            group_period = "day"
        else:
            group_period = "month"

        da = da.assign_coords(
            {
                group_period: getattr(da[time_dim].dt, group_period),
                "hour": da[time_dim].dt.hour,
            }
        )

        count_summary = None

        if check_group_counts:
            time_groups = da[time_dim].assign_coords(
                {
                    group_period: getattr(da[time_dim].dt, group_period),
                    "hour": da[time_dim].dt.hour,
                }
            )

            counts = time_groups.groupby([group_period, "hour"]).count()

            count_summary = _group_count_summary(
                counts,
                group_dims=(group_period, "hour"),
            )

        clim = da.groupby([group_period, "hour"]).mean(time_dim)

        if count_summary is not None:
            clim.attrs["climatology_group_counts"] = count_summary

        clim = clim.chunk({group_period: -1})

        if rolling_window is not None:
            if rolling_window > clim.sizes[group_period]:
                raise ValueError(
                    f"rolling_window={rolling_window} is larger than "
                    f"{group_period} size={clim.sizes[group_period]} "
                    f"for clim_period={clim_period}"
                )

            clim = clim.rolling(
                {group_period: rolling_window},
                center=rolling_center,
                min_periods=rolling_min_periods,
            ).mean()

        return cast(T_Xarray, clim)

    count_summary = None

    if check_group_counts:
        counts = da[time_dim].groupby(f"{time_dim}.{clim_period}").count()
        counts = counts.compute() if hasattr(counts.data, "compute") else counts

        count_summary = {
            "sizes": dict(counts.sizes),
            "min_count": int(counts.min()),
            "max_count": int(counts.max()),
        }

    clim = da.groupby(f"{time_dim}.{clim_period}").mean(time_dim)

    if count_summary is not None:
        clim.attrs["climatology_group_counts"] = count_summary

    if rolling_window is not None:
        if rolling_window > clim.sizes[clim_period]:
            raise ValueError(
                f"rolling_window={rolling_window} is larger than "
                f"{clim_period} size={clim.sizes[clim_period]}"
            )

        clim = clim.rolling(
            {clim_period: rolling_window},
            center=rolling_center,
            min_periods=rolling_min_periods,
        ).mean()

    return cast(T_Xarray, clim)


def calculate_save_and_subset_climatologies(
    s: Settings,
    leadtime_units: LeadtimeUnit,
    force: bool = False,
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    rolling_window: int | None = None,
    rolling_center: bool = True,
    rolling_min_periods: int = 1,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_range: tuple[str, str] | None = None,
    time_start: str | None = None,
    interpolate: bool = True,
    engine: Literal["netcdf", "zarr"] = "zarr",
    build_analysis: bool = True,
    coord_rename_fc: Sequence[Sequence[str]] | None = None,
    coord_rename_an: Sequence[Sequence[str]] | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset | None]:
    clim_time_range = time_range or (s.train_start, s.train_end)
    region_label = s.region_name or "global"

    start_label = pd.Timestamp(clim_time_range[0]).strftime("%Y%m%d")
    end_label = pd.Timestamp(clim_time_range[1]).strftime("%Y%m%d")

    clim_label = f"{clim_period}_{start_label}_{end_label}"

    # Original forecast climatology
    fc_path = s.input_fc
    fc_clim_path = (
        s.input_clim_dir
        / (
            f"{s.model_fc}_{s.var_fc}_{region_label}_"
            f"{clim_label}_climatology.zarr"
        )
    )

    # Analysis climatology
    an_path = s.input_an
    an_clim_path = (
        s.input_clim_dir
        / (
            f"{s.model_an}_{s.var_an}_{region_label}_"
            f"{clim_label}_climatology.zarr"
        )
    )

    if not fc_path.exists():
        raise FileNotFoundError(f"Couldn't find forecast dataset {fc_path}")
    if not an_path.exists():
        raise FileNotFoundError(f"Couldn't find analysis dataset {an_path}")

    # Corrected forecast climatology
    train_pred_path = s.output_dir / "train_corrected.zarr"
    train_pred_clim_path = (
        s.output_clim_dir
        / f"train_corrected_{clim_label}_climatology.zarr"
    )

    s.make_dirs()

    need_base_clim = force or not fc_clim_path.exists() or not an_clim_path.exists()
    need_mlfc_clim = force or not train_pred_clim_path.exists()

    need_input_data = need_base_clim or need_mlfc_clim

    if need_input_data:
        fc, an, mlfc = get_and_subset_datasets(
            s,
            leadtime_units=leadtime_units,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            time_start=time_start,
            interpolate=interpolate,
            engine=engine,
            build_analysis=build_analysis,
            coord_rename_fc=coord_rename_fc,
            coord_rename_an=coord_rename_an,
        )
        fc = fc[s.var_file_fc]
        an = an[s.var_file_an]
        mlfc = mlfc[s.var_file_fc] if mlfc is not None else None

    mlfc_clim = None
    if train_pred_path.exists():
        if need_mlfc_clim:
            print("Save ML-corrected forecast climatology to:", train_pred_clim_path.name)

            train_pred = open_zarr(train_pred_path)

            with ProgressBar():
                mlfc_clim = calculate_climatology(
                    train_pred[s.var_fc],
                    time_dim=train_pred.earthml.guessed_dims.time,
                    clim_period=clim_period,
                    rolling_window=rolling_window,
                    rolling_center=rolling_center,
                    rolling_min_periods=rolling_min_periods,
                ).compute()

            mlfc_clim_ds = mlfc_clim.to_dataset(name=s.var_file_fc)
            mlfc_clim_ds = save_zarr(
                mlfc_clim_ds,
                train_pred_clim_path,
                safe_chunk_spec(mlfc_clim_ds, train_pred.unify_chunks()),
            )
            mlfc_clim = mlfc_clim_ds[s.var_fc]
        else:
            mlfc_clim = open_zarr_var(train_pred_clim_path, s.var_fc)
    else:
        print(f"Skipping ML-corrected forecast climatology")

    should_compute = force or not fc_clim_path.exists() or not an_clim_path.exists()
    if need_base_clim:
        print("Save original forecast climatology to:", fc_clim_path.name)

        fc_train = fc.sel({fc.earthml.guessed_dims.time: slice(*clim_time_range)})

        with ProgressBar():
            print("Calculate original forecast climatology")
            fc_clim = calculate_climatology(
                fc_train,
                time_dim=fc_train.earthml.guessed_dims.time,
                clim_period=clim_period,
                rolling_window=rolling_window,
                rolling_center=rolling_center,
                rolling_min_periods=rolling_min_periods,
            ).compute()

        fc_clim_ds = fc_clim.to_dataset(name=s.var_file_fc)
        fc_clim_ds = save_zarr(
            fc_clim_ds,
            fc_clim_path,
            safe_chunk_spec(fc_clim_ds, fc_train.to_dataset(name=s.var_file_fc).unify_chunks()),
        )
        fc_clim = fc_clim_ds[s.var_fc]

        print("Save analysis climatology to:", an_clim_path.name)

        an_train = an.sel({an.earthml.guessed_dims.time: slice(*clim_time_range)})

        with ProgressBar():
            print("Calculate analysis climatology")
            an_clim = calculate_climatology(
                an_train,
                time_dim=an_train.earthml.guessed_dims.time,
                clim_period=clim_period,
                rolling_window=rolling_window,
                rolling_center=rolling_center,
                rolling_min_periods=rolling_min_periods,
            ).compute()

        an_clim_ds = an_clim.to_dataset(name=s.var_file_an)
        an_clim_ds = save_zarr(
            an_clim_ds,
            an_clim_path,
            safe_chunk_spec(an_clim_ds, an_train.to_dataset(name=s.var_file_an).unify_chunks()),
        )
        an_clim = an_clim_ds[s.var_an]

    else:
        fc_clim = open_zarr_var(fc_clim_path, s.var_file_fc)
        an_clim = open_zarr_var(an_clim_path, s.var_file_an)


    lon_dim = fc_clim.earthml.guessed_dims.longitude
    lat_dim = fc_clim.earthml.guessed_dims.latitude

    if interpolate:
        an_clim = an_clim.interp(
            {
                lat_dim: fc_clim[lat_dim],
                lon_dim: fc_clim[lon_dim],
            }
        )

    return (
        subset_dataset(
            fc_clim.to_dataset(name=s.var_file_fc),
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=None,
            time_start=time_start,
        ),
        subset_dataset(
            an_clim.to_dataset(name=s.var_file_an),
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=None,
            time_start=time_start,
        ),
        subset_dataset(
            mlfc_clim.to_dataset(name=s.var_file_fc),
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=None,
            time_start=time_start,
        ) if mlfc_clim is not None else None,
    )


def select_clim_for_time(
    clim: xr.Dataset,
    times,
    clim_period: ClimPeriod,
    time_dim: str = "time",
) -> xr.Dataset:
    times = pd.DatetimeIndex(times)

    if clim_period == "month":
        return clim.sel(
            month=xr.DataArray(
                times.month,
                dims=time_dim,
                coords={time_dim: times},
            )
        )

    if clim_period == "dayofyear":
        return clim.sel(
            dayofyear=xr.DataArray(
                times.dayofyear,
                dims=time_dim,
                coords={time_dim: times},
            )
        )

    if clim_period == "day":
        return clim.sel(
            day=xr.DataArray(
                times.day,
                dims=time_dim,
                coords={time_dim: times},
            )
        )

    if clim_period == "hour":
        return clim.sel(
            hour=xr.DataArray(
                times.hour,
                dims=time_dim,
                coords={time_dim: times},
            )
        )

    if clim_period in {"dayofyear_hour", "day_hour", "month_hour"}:
        if clim_period == "dayofyear_hour":
            group_period = "dayofyear"
            group_values = times.dayofyear
        elif clim_period == "day_hour":
            group_period = "day"
            group_values = times.day
        else:
            group_period = "month"
            group_values = times.month

        return clim.sel(
            {
                group_period: xr.DataArray(
                    group_values,
                    dims=time_dim,
                    coords={time_dim: times},
                ),
                "hour": xr.DataArray(
                    times.hour,
                    dims=time_dim,
                    coords={time_dim: times},
                ),
            }
        )

    raise ValueError(f"Unsupported clim_period={clim_period!r}")


def _group_count_summary(counts: xr.DataArray, group_dims: tuple[str, ...]) -> dict:
    counts_1d = counts.isel(
        {dim: 0 for dim in counts.dims if dim not in group_dims}
    ).compute()

    max_count = int(counts_1d.max())
    min_count = int(counts_1d.min())

    summary = {
        "min_count": min_count,
        "max_count": max_count,
        "sizes": dict(counts_1d.sizes),
    }

    bad = counts_1d.where(counts_1d != max_count, drop=True)

    if bad.size:
        bad_groups = []

        for count_value in np.unique(bad.values[np.isfinite(bad.values)]):
            subset = bad.where(bad == count_value, drop=True)

            bad_groups.append(
                {
                    "count": int(count_value),
                    **{
                        dim: subset[dim].values.tolist()
                        for dim in group_dims
                        if dim in subset.coords
                    },
                }
            )

        summary["low_count_groups"] = bad_groups

    return summary
