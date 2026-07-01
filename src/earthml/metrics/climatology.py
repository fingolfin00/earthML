from typing import Sequence, Literal, TypeVar, cast

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dask.diagnostics.progress import ProgressBar

from ..base import (
    Settings,
    open_zarr,
    open_zarr_var,
    open_nc,
    open_nc_var,
    T_Xarray,
    select_target_for_lead,
    subset_dataset,
    get_and_subset_datasets,
)


def calculate_climatology(
    da: xr.DataArray | xr.Dataset,
    time_dim = "time",
    clim_period: str = "month",
) -> T_Xarray:
    return cast(T_Xarray, da.groupby(f"{time_dim}.{clim_period}").mean(time_dim))


def calculate_save_and_subset_climatologies(
    s: Settings,
    period: Literal["hours", "days", "months", "years"],
    force: bool = False,
    clim_period: Literal["day", "month", "year"] = "month",
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

    # Original forecast climatology
    fc_path = s.input_fc
    fc_clim_path = s.input_clim_dir / f"{s.model_fc}_{s.var_fc}_train_clim.zarr"

    # Analysis climatology
    an_path = s.input_an
    an_clim_path = s.input_clim_dir / f"{s.model_an}_{s.var_an}_train_clim.zarr"

    if not fc_path.exists():
        raise FileNotFoundError(f"Couldn't find forecast dataset {fc_path}")
    if not an_path.exists():
        raise FileNotFoundError(f"Couldn't find analysis dataset {an_path}")

    # Corrected forecast climatology
    train_pred_path = s.output_dir / "train_corrected.zarr"
    train_pred_clim_path = s.output_clim_dir / "train_corrected_clim.zarr"

    print(f"Get climatologies ({clim_time_range[0]} - {clim_time_range[1]}) for experiment {s.output_name}")

    s.make_dirs()

    need_base_clim = force or not fc_clim_path.exists() or not an_clim_path.exists()
    need_mlfc_clim = force or not train_pred_clim_path.exists()

    need_input_data = need_base_clim or need_mlfc_clim

    if need_input_data:
        fc, an, mlfc = get_and_subset_datasets(
            s,
            period=period,
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

            train_pred = open_zarr_var(train_pred_path, s.var_fc)

            with ProgressBar():
                mlfc_clim = calculate_climatology(train_pred, time_dim=train_pred.earthml.guessed_dims.time, clim_period=clim_period).compute()

            mlfc_clim.to_dataset(name=s.var_file_fc).to_zarr(
                train_pred_clim_path,
                mode="w",
                consolidated=False,
                zarr_format=2,
            )
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
            fc_clim = calculate_climatology(fc_train).compute()

        fc_clim.to_dataset(name=s.var_file_fc).to_zarr(
            fc_clim_path,
            mode="w",
            consolidated=False,
            zarr_format=2,
        )

        print("Save analysis climatology to:", an_clim_path.name)

        an_train = an.sel({fc.earthml.guessed_dims.time: slice(*clim_time_range)})

        with ProgressBar():
            print("Calculate original forecast climatology")
            an_clim = calculate_climatology(an_train).compute()

        an_clim.to_dataset(name=s.var_file_an).to_zarr(
            an_clim_path,
            mode="w",
            consolidated=False,
            zarr_format=2,
        )
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
