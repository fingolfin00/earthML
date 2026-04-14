from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/earthml-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import welch

from ...metrics import CorrelationMetrics, DeterministicMetrics
from ...plots import plot_realization_timeseries, plot_temporal_mean_map, build_plot_config, get_var_units
from ...plots.timeseries import _reduce_for_timeseries


PlotSpec = dict[str, Any]
DEFAULT_PLOT_CONFIG = build_plot_config()
BOOKKEEPING_VARS = {"_has_var"}
METRIC_SPECS = [
    {"name": "rmse", "unit_mode": "base"},
    {"name": "mae", "unit_mode": "base"},
    {"name": "crmse", "unit_mode": "base"},
    {"name": "bias", "unit_mode": "base"},
    {"name": "nrmse", "unit_mode": None},
    {"name": "nmae", "unit_mode": None},
    {"name": "nbias", "unit_mode": None},
    {"name": "corr", "unit_mode": None},
    {"name": "clim_anom_corr", "unit_mode": None},
    {"name": "spatial_anom_corr", "unit_mode": None},
]


def _get_stage_plot_folder(root: Path, data_type: str) -> Path:
    stage_plot_folder = root.joinpath(data_type)
    stage_plot_folder.mkdir(parents=True, exist_ok=True)
    return stage_plot_folder


def _sanitize_plot_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def export_lightning_curves(
    *,
    logger,
    log_dir: str | Path,
    plots_folder_path: Path,
) -> None:
    log_dir = Path(log_dir)
    if not log_dir.exists():
        logger.warning("Skip Lightning curve export: log dir does not exist: %s", log_dir)
        return

    training_curves_folder_path = plots_folder_path.joinpath("training_curves")
    training_curves_folder_path.mkdir(parents=True, exist_ok=True)

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("Skip Lightning curve export: tensorboard is not installed.")
        return

    try:
        event_accumulator = EventAccumulator(str(log_dir))
        event_accumulator.Reload()
        scalar_tags = event_accumulator.Tags().get("scalars", [])
    except Exception as exc:
        logger.warning("Skip Lightning curve export: failed to read TensorBoard events in %s (%s)", log_dir, exc)
        return

    if not scalar_tags:
        logger.info("Skip Lightning curve export: no scalar tags found in %s", log_dir)
        return

    exported = 0
    for tag in scalar_tags:
        try:
            scalar_events = event_accumulator.Scalars(tag)
        except Exception as exc:
            logger.warning("Skip Lightning scalar tag %s: %s", tag, exc)
            continue

        if not scalar_events:
            continue

        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(steps, values, linewidth=2)
        ax.set_title(tag.replace("_", " "))
        ax.set_xlabel("step")
        ax.set_ylabel(tag)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = training_curves_folder_path / f"{_sanitize_plot_name(tag)}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        exported += 1

    if exported == 0:
        logger.info("Skip Lightning curve export: scalar tags were found in %s but no plots were written.", log_dir)
        return

    logger.info(
        "Saved %s Lightning curve plot(s) to %s",
        exported,
        training_curves_folder_path,
    )


def _select_plot_var(ds: xr.Dataset, var: str) -> xr.Dataset:
    return ds[[var]] if len(ds.data_vars) > 1 else ds


def _data_vars_for_plotting(ds: xr.Dataset) -> list[str]:
    return [var for var in ds.data_vars if var not in BOOKKEEPING_VARS]


def _resolve_plot_var_name(ds: xr.Dataset, var: str) -> str:
    if var in ds.data_vars:
        return var
    plot_vars = _data_vars_for_plotting(ds)
    if len(plot_vars) == 1:
        return plot_vars[0]
    raise KeyError(f"Variable '{var}' not found in dataset with variables {list(ds.data_vars)}.")


def _select_plot_var_flexible(ds: xr.Dataset, var: str) -> xr.Dataset:
    return ds[[_resolve_plot_var_name(ds, var)]]


def _sanitize_plot_time_coords(ds: xr.Dataset, *, x_dim: str) -> xr.Dataset:
    """
    Remove auxiliary time-like coordinates that can visually conflict with the
    canonical plotting axis.

    Earthkit datasets may carry `source_time` and `valid_time` alongside the
    canonical `time` dimension. Diagnostic plots should always use the
    canonical axis only.
    """
    if x_dim != "time":
        return ds

    drop_coords = [
        name
        for name in ("source_time", "valid_time")
        if name in ds.coords and name != x_dim
    ]
    if not drop_coords:
        return ds
    return ds.drop_vars(drop_coords)


def _get_plot_members(ds: xr.Dataset) -> tuple[xr.Dataset | None, str | None]:
    rdim = ds.earthml.guessed_dims.realization
    if rdim is not None and rdim in ds.dims:
        return ds, rdim
    return None, rdim


def _get_var_unit(ds: xr.Dataset, var: str) -> str | None:
    ds_var = _select_plot_var_flexible(ds, var)
    da = next(iter(ds_var.data_vars.values()))
    unit = da.attrs.get("units")
    if unit:
        return unit
    try:
        return get_var_units(DEFAULT_PLOT_CONFIG, var)
    except KeyError:
        return None


def _format_y_label(var: str, *, unit: str | None, quantity: str = "") -> str:
    if quantity:
        return f"{quantity} [{unit}]" if unit else quantity
    return f"{var} [{unit}]" if unit else var


def _square_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    return f"{unit}^2"


def _infer_period_unit(ds: xr.Dataset) -> str:
    time_dim = ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in ds.coords:
        return "timesteps"

    time_vals = ds[time_dim].values
    if len(time_vals) < 2:
        return "timesteps"

    try:
        time_idx = xr.DataArray(time_vals, dims=(time_dim,)).to_index()
    except Exception:
        return "timesteps"

    try:
        freq = getattr(time_idx, "freqstr", None) or xr.infer_freq(time_idx)
    except Exception:
        freq = None

    if freq is not None:
        freq = str(freq).upper()
        if "MS" in freq or freq == "M":
            return "months"
        if "D" in freq:
            return "days"
        if "H" in freq:
            return "hours"

    try:
        diffs = np.diff(time_vals.astype("datetime64[ns]")).astype("timedelta64[s]").astype(np.int64)
        if diffs.size == 0:
            return "timesteps"
        median_seconds = int(np.median(diffs))
    except Exception:
        return "timesteps"

    day = 24 * 3600
    if 27 * day <= median_seconds <= 32 * day:
        return "months"
    if median_seconds % day == 0:
        return "days"
    if median_seconds % 3600 == 0:
        return "hours"
    return "timesteps"


def _extract_requested_leadtime(ds: xr.Dataset) -> np.timedelta64 | None:
    if "leadtime" in ds.coords and ds["leadtime"].size == 1:
        try:
            return np.asarray(ds["leadtime"].values).reshape(-1)[0]
        except Exception:
            pass

    removed = ds.attrs.get("removed_metadata")
    if isinstance(removed, dict):
        lead_meta = removed.get("leadtime")
        if isinstance(lead_meta, dict):
            value = lead_meta.get("value")
            if isinstance(value, list) and len(value) == 1:
                try:
                    return np.asarray(value, dtype="timedelta64[ns]")[0]
                except Exception:
                    pass

    return None


def _infer_lag_steps(ds: xr.Dataset, *, time_dim: str) -> int:
    leadtime = _extract_requested_leadtime(ds)
    if leadtime is None:
        return 1

    if time_dim not in ds.coords or ds.sizes.get(time_dim, 0) < 2:
        return 1

    try:
        time_vals = np.asarray(ds[time_dim].values).astype("datetime64[ns]")
        deltas = np.diff(time_vals).astype("timedelta64[ns]").astype(np.int64)
        if deltas.size == 0:
            return 1
        dt_ns = int(np.median(deltas))
        lt_ns = int(np.asarray(leadtime).astype("timedelta64[ns]").astype(np.int64))
        if dt_ns <= 0:
            return 1
        return max(1, int(round(lt_ns / dt_ns)))
    except Exception:
        return 1


def _format_map_cbar_label(var: str, unit: str | None) -> str:
    return f"{var} [{unit}]" if unit else var


def _metric_display_name(metric_name: str) -> str:
    return metric_name.replace("_", " ")


def _metric_unit(metric_name: str, base_unit: str | None) -> str | None:
    metric_spec = next((spec for spec in METRIC_SPECS if spec["name"] == metric_name), None)
    if metric_spec is None:
        return None
    return base_unit if metric_spec["unit_mode"] == "base" else None


def _format_metric_label(metric_name: str, *, unit: str | None) -> str:
    return _format_y_label(_metric_display_name(metric_name), unit=unit)


def _metric_model_specs(
    plot_specs: list[PlotSpec],
    truth_ds: xr.Dataset,
) -> list[PlotSpec]:
    return [spec for spec in plot_specs if spec["ds"] is not truth_ds]


def _build_metric_datasets(
    *,
    truth_ds: xr.Dataset,
    model_specs: list[PlotSpec],
    reduce_dims: tuple[str, ...],
) -> dict[str, xr.Dataset]:
    if not model_specs or not reduce_dims:
        return {}

    model_data = [spec["ds"] for spec in model_specs]
    model_names = [spec["label"] for spec in model_specs]

    det_metrics = DeterministicMetrics(
        truth_data=truth_ds,
        model_data=model_data,
        truth_name="target",
        model_names=model_names,
    )
    corr_metrics = CorrelationMetrics(
        truth_data=truth_ds,
        model_data=model_data,
        truth_name="target",
        model_names=model_names,
        clim_data=[_build_climatology_ds(truth_ds)] + [_build_climatology_ds(ds) for ds in model_data],
    )

    return {
        "rmse": det_metrics.rmse(reduce_dims),
        "mae": det_metrics.mae(reduce_dims),
        "crmse": det_metrics.crmse(reduce_dims),
        "bias": det_metrics.bias(reduce_dims),
        "nrmse": det_metrics.nrmse(reduce_dims),
        "nmae": det_metrics.nmae(reduce_dims),
        "nbias": det_metrics.nbias(reduce_dims),
        "corr": corr_metrics.corr(reduce_dims),
        "clim_anom_corr": corr_metrics.clim_anom_corr(reduce_dims),
        "spatial_anom_corr": corr_metrics.spatial_anom_corr(reduce_dims),
    }


def _select_metric_model_ds(metric_ds: xr.Dataset, model_name: str) -> xr.Dataset:
    return metric_ds.sel(model=model_name, drop=True)


def _scalar_pair_metrics(
    *,
    truth: xr.DataArray,
    model: xr.DataArray,
    reduce_dim: str,
) -> tuple[float, float]:
    var_name = truth.name or model.name or "value"
    truth_ds = truth.to_dataset(name=var_name)
    model_ds = model.to_dataset(name=var_name)

    rmse = DeterministicMetrics(
        truth_data=truth_ds,
        model_data=model_ds,
        truth_name="truth",
        model_names="model",
    ).rmse(reduce_dim)[var_name].sel(model="model", drop=True)
    corr = CorrelationMetrics(
        truth_data=truth_ds,
        model_data=model_ds,
        truth_name="truth",
        model_names="model",
    ).corr(reduce_dim)[var_name].sel(model="model", drop=True)

    return float(rmse.values), float(corr.values)


def _plot_single_timeseries(
    *,
    logger,
    ax,
    ds: xr.Dataset,
    var: str,
    time_dim: str,
    stage_kind: str,
    data_type: str,
    label: str,
    mean_label: str,
    color: str,
    y_label: str | None = None,
) -> None:
    ds_var = _select_plot_var(ds, var)
    members, rdim = _get_plot_members(ds_var)
    logger.info(
        "Plot %s %s/%s: %s realizations=%s",
        stage_kind,
        data_type,
        var,
        label,
        ds_var.sizes.get(rdim, 0) if members is not None and rdim is not None else 0,
    )
    plot_realization_timeseries(
        ds_var,
        members=members,
        x_dim=time_dim,
        ens_dim=rdim or "realization",
        ax=ax,
        x_label="Time",
        y_label=y_label,
        label=label,
        mean_label=mean_label,
        color=color,
        spread="minmax",
    )


def _build_residual_ds(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    var: str,
) -> xr.Dataset:
    left_var = _select_plot_var_flexible(left_ds, var)
    right_var = _select_plot_var_flexible(right_ds, var)

    rdim_left = left_var.earthml.guessed_dims.realization
    rdim_right = right_var.earthml.guessed_dims.realization

    # If the reference side is deterministic/singleton while the left side is
    # ensemble, drop the singleton realization axis so xarray broadcasts it
    # instead of intersecting realization coordinates during subtraction.
    if (
        rdim_left is not None
        and rdim_right is not None
        and rdim_left == rdim_right
        and left_var.sizes.get(rdim_left, 1) > 1
        and right_var.sizes.get(rdim_right, 1) == 1
    ):
        right_var = right_var.squeeze(rdim_right, drop=True)
        rdim_right = None

    exclude_align_dims = tuple(dim for dim in (rdim_left, rdim_right) if dim is not None)
    left_var, right_var = xr.align(left_var, right_var, join="inner", exclude=exclude_align_dims)

    return left_var - right_var


def _build_climatology_ds(ds: xr.Dataset) -> xr.Dataset:
    return ds.earthml.climatology(groupby="month")


def _build_ds_minus_climatology_ds(ds: xr.Dataset) -> xr.Dataset:
    time_dim = ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in ds.dims:
        raise ValueError("Cannot compute dataset minus climatology without a time dimension.")
    clim_ds = _build_climatology_ds(ds)
    return ds.groupby(f"{time_dim}.month") - clim_ds


def _build_anomaly_residual_ds(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    var: str,
) -> xr.Dataset:
    left_anom = _build_ds_minus_climatology_ds(left_ds)
    right_anom = _build_ds_minus_climatology_ds(right_ds)
    return _build_residual_ds(left_anom, right_anom, var=var)


def _build_anomaly_variance_ds(ds: xr.Dataset, window: int | None = None) -> xr.Dataset:
    time_dim = ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in ds.dims:
        raise ValueError("Cannot compute anomaly variance without a time dimension.")

    anom_ds = _build_ds_minus_climatology_ds(ds)
    time_len = int(anom_ds.sizes.get(time_dim, 0))
    if time_len < 3:
        raise ValueError("Need at least 3 timesteps to compute anomaly variance timeseries.")

    window = min(12, time_len) if window is None else min(window, time_len)
    min_periods = max(3, window // 2)

    return anom_ds.rolling({time_dim: window}, center=True, min_periods=min_periods).var()


def _build_anomaly_autocorr_ds(ds: xr.Dataset, nlags: int | None = None) -> xr.Dataset:
    time_dim = ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in ds.dims:
        raise ValueError("Cannot compute anomaly autocorrelation without a time dimension.")

    anom_ds = _build_ds_minus_climatology_ds(ds)
    time_len = int(anom_ds.sizes.get(time_dim, 0))
    if time_len < 3:
        raise ValueError("Need at least 3 timesteps to compute anomaly autocorrelation.")

    nlags = min(12, time_len - 1) if nlags is None else min(nlags, time_len - 1)
    if nlags < 1:
        raise ValueError("Need at least 1 valid lag to compute anomaly autocorrelation.")

    acf_vars: dict[str, xr.DataArray] = {}
    for var_name, da in anom_ds.data_vars.items():
        corr_list: list[xr.DataArray] = []
        for lag in range(nlags + 1):
            da_lag = da.shift({time_dim: lag})
            x = da
            y = da_lag
            x_mean = x.mean(dim=time_dim, skipna=True)
            y_mean = y.mean(dim=time_dim, skipna=True)
            x_anom = x - x_mean
            y_anom = y - y_mean
            cov = (x_anom * y_anom).mean(dim=time_dim, skipna=True)
            x_std = x.std(dim=time_dim, skipna=True)
            y_std = y.std(dim=time_dim, skipna=True)
            denom = x_std * y_std
            valid = np.isfinite(denom) & (denom != 0)
            denom_safe = denom.where(valid, other=1.0)
            corr = (cov / denom_safe).where(valid)
            corr_list.append(corr)
        acf_vars[var_name] = xr.concat(
            corr_list,
            dim=xr.IndexVariable("lag", range(nlags + 1)),
        )

    return xr.Dataset(acf_vars)


def _build_anomaly_power_spectrum_ds(ds: xr.Dataset) -> xr.Dataset:
    time_dim = ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in ds.dims:
        raise ValueError("Cannot compute anomaly power spectrum without a time dimension.")

    anom_ds = _build_ds_minus_climatology_ds(ds)
    time_len = int(anom_ds.sizes.get(time_dim, 0))
    if time_len < 3:
        raise ValueError("Need at least 3 timesteps to compute anomaly power spectrum.")

    # Welch operates along the full time axis, so make the core dimension a
    # single chunk when working with Dask-backed arrays.
    try:
        chunks = anom_ds.chunks
    except ValueError:
        anom_ds = anom_ds.unify_chunks()
        chunks = anom_ds.chunks
    if chunks is not None:
        anom_ds = anom_ds.chunk({time_dim: -1})

    nperseg = min(12, time_len)
    if nperseg < 3:
        raise ValueError("Need at least 3 samples per segment to compute Welch spectrum.")
    noverlap = min(nperseg // 2, nperseg - 1)
    nfreq = nperseg // 2 + 1

    psd_vars: dict[str, xr.DataArray] = {}
    for var_name, da in anom_ds.data_vars.items():
        freq_ref, _ = welch(
            np.zeros(time_len, dtype=np.float64),
            fs=1.0,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            axis=-1,
        )
        spectrum = xr.apply_ufunc(
            lambda x: welch(
                x,
                fs=1.0,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                scaling="density",
                axis=-1,
            )[1],
            da.transpose(..., time_dim),
            input_core_dims=[[time_dim]],
            output_core_dims=[["frequency"]],
            exclude_dims={time_dim},
            vectorize=False,
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={
                "output_sizes": {"frequency": nfreq},
                "allow_rechunk": True,
            },
        )
        period_ref = np.where(freq_ref > 0, 1.0 / freq_ref, np.inf)
        psd_vars[var_name] = spectrum.assign_coords(frequency=("frequency", period_ref))

    return xr.Dataset(psd_vars)


def plot_stage_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
    x_dim: str | None = None,
    x_label: str = "Time",
    filename_suffix: str = "timeseries",
    y_label_mode: str = "unit",
) -> None:
    stage_plot_folder = _get_stage_plot_folder(plots_folder_path, data_type)
    logger.info(
        "Generate %s stage plots for %s (%s) in %s",
        stage_kind,
        data_type,
        stage,
        stage_plot_folder,
    )

    base_ds = plot_specs[0]["ds"]
    plot_dim = x_dim or base_ds.earthml.guessed_dims.time
    if plot_dim is None or any(plot_dim not in spec["ds"].dims for spec in plot_specs):
        logger.debug("Skip %s plotting for %s: no common plot dimension.", stage_kind, data_type)
        return

    plot_specs = [
        {
            **spec,
            "ds": _sanitize_plot_time_coords(spec["ds"], x_dim=plot_dim),
        }
        for spec in plot_specs
    ]
    base_ds = plot_specs[0]["ds"]

    common_vars = [
        var for var in _data_vars_for_plotting(plot_specs[0]["ds"])
        if all(var in spec["ds"].data_vars or len(_data_vars_for_plotting(spec["ds"])) == 1 for spec in plot_specs[1:])
    ]
    if not common_vars:
        logger.debug("Skip %s plotting for %s: no common variables.", stage_kind, data_type)
        return

    for var in common_vars:
        fig, ax = plt.subplots(figsize=(10, 4))
        try:
            base_unit = _get_var_unit(base_ds, var)
            if y_label_mode == "unit":
                y_label = _format_y_label(var, unit=base_unit)
            elif y_label_mode == "variance":
                y_label = _format_y_label(var, unit=_square_unit(base_unit), quantity="Variance")
            elif y_label_mode == "autocorr":
                y_label = "Autocorrelation"
            elif y_label_mode == "power_spectrum":
                y_label = _format_y_label(var, unit=_square_unit(base_unit), quantity="Power spectral density")
            else:
                y_label = None
            for spec in plot_specs:
                ds_var = _select_plot_var_flexible(spec["ds"], var)
                members, rdim = _get_plot_members(ds_var)
                logger.info(
                    "Plot %s %s/%s: %s realizations=%s",
                    stage_kind,
                    data_type,
                    var,
                    spec["label"],
                    ds_var.sizes.get(rdim, 0) if members is not None and rdim is not None else 0,
                )
                plot_realization_timeseries(
                    ds_var,
                    members=members,
                    x_dim=plot_dim,
                    ens_dim=rdim or "realization",
                    ax=ax,
                    x_label=x_label,
                    y_label=y_label,
                    label=spec["label"],
                    mean_label=spec["mean_label"],
                    color=spec["color"],
                )

            ax.legend()
            fig.tight_layout()
            fig.savefig(stage_plot_folder.joinpath(f"{stage}_{var}_{filename_suffix}.png"), dpi=200)
        except Exception as exc:
            logger.warning(
                "Failed to generate %s plot for %s/%s/%s: %s",
                stage_kind,
                data_type,
                stage,
                var,
                repr(exc),
            )
        finally:
            plt.close(fig)


def plot_stage_residual_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    data_type: str,
    stage: str,
    stage_kind: str,
    residual_label: str,
    residual_mean_label: str,
    color: str = "tab:red",
    anomaly: bool = False,
) -> None:
    stage_plot_folder = _get_stage_plot_folder(plots_folder_path, data_type)
    logger.info(
        "Generate %s residual plots for %s (%s) in %s",
        stage_kind,
        data_type,
        stage,
        stage_plot_folder,
    )

    time_dim = left_ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in left_ds.dims or time_dim not in right_ds.dims:
        logger.debug("Skip %s residual plotting for %s: no common time dimension.", stage_kind, data_type)
        return

    left_ds = _sanitize_plot_time_coords(left_ds, x_dim=time_dim)
    right_ds = _sanitize_plot_time_coords(right_ds, x_dim=time_dim)

    common_vars = [
        var for var in _data_vars_for_plotting(left_ds)
        if var in right_ds.data_vars or len(_data_vars_for_plotting(right_ds)) == 1
    ]
    if not common_vars:
        logger.debug("Skip %s residual plotting for %s: no common variables.", stage_kind, data_type)
        return

    for var in common_vars:
        fig, ax = plt.subplots(figsize=(10, 4))
        try:
            residual_ds = (
                _build_anomaly_residual_ds(left_ds, right_ds, var=var)
                if anomaly else
                _build_residual_ds(left_ds, right_ds, var=var)
            )
            _plot_single_timeseries(
                logger=logger,
                ax=ax,
                ds=residual_ds,
                var=var,
                time_dim=time_dim,
                stage_kind=f"{stage_kind} residual",
                data_type=data_type,
                label=residual_label,
                mean_label=residual_mean_label,
                color=color,
                y_label=_format_y_label(var, unit=_get_var_unit(left_ds, var)),
            )
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
            ax.legend()
            fig.tight_layout()
            suffix = "anomaly_residual_timeseries" if anomaly else "raw_residual_timeseries"
            fig.savefig(stage_plot_folder.joinpath(f"{stage}_{var}_{suffix}.png"), dpi=200)
        except Exception as exc:
            logger.warning(
                "Failed to generate %s residual plot for %s/%s/%s: %s",
                stage_kind,
                data_type,
                stage,
                var,
                repr(exc),
            )
        finally:
            plt.close(fig)


def plot_stage_lag_diagnostic(
    *,
    logger,
    plots_folder_path: Path,
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    data_type: str,
    stage: str,
    stage_kind: str,
    lag_steps: int | None = None,
) -> None:
    stage_plot_folder = _get_stage_plot_folder(plots_folder_path, data_type)
    time_dim = left_ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in left_ds.dims or time_dim not in right_ds.dims:
        logger.debug("Skip %s lag diagnostic for %s: no common time dimension.", stage_kind, data_type)
        return

    left_ds = _sanitize_plot_time_coords(left_ds, x_dim=time_dim)
    right_ds = _sanitize_plot_time_coords(right_ds, x_dim=time_dim)

    common_vars = [
        var for var in _data_vars_for_plotting(left_ds)
        if var in right_ds.data_vars or len(_data_vars_for_plotting(right_ds)) == 1
    ]
    if not common_vars:
        return

    for var in common_vars:
        fig, ax = plt.subplots(figsize=(10, 4))
        try:
            left_var = _select_plot_var_flexible(left_ds, var)
            right_var = _select_plot_var_flexible(right_ds, var)

            input_mean = _reduce_for_timeseries(left_var, keep_dims=(time_dim,))
            input_mean = input_mean.transpose(time_dim)

            target_mean = _reduce_for_timeseries(right_var, keep_dims=(time_dim,))
            target_mean = target_mean.transpose(time_dim)

            input_mean, target_mean = xr.align(input_mean, target_mean, join="inner")
            x = input_mean[time_dim].values

            lag_steps_eff = lag_steps if lag_steps is not None else _infer_lag_steps(left_ds, time_dim=time_dim)
            target_prev = target_mean.shift({time_dim: lag_steps_eff})
            target_next = target_mean.shift({time_dim: -lag_steps_eff})

            rmse_0, corr_0 = _scalar_pair_metrics(truth=target_mean, model=input_mean, reduce_dim=time_dim)
            rmse_prev, corr_prev = _scalar_pair_metrics(truth=target_prev, model=input_mean, reduce_dim=time_dim)
            rmse_next, corr_next = _scalar_pair_metrics(truth=target_next, model=input_mean, reduce_dim=time_dim)

            logger.info(
                "Lag diagnostic %s/%s/%s: lag0 rmse=%.3f corr=%.3f | target(t-%s) rmse=%.3f corr=%.3f | target(t+%s) rmse=%.3f corr=%.3f",
                data_type,
                stage,
                var,
                rmse_0,
                corr_0,
                lag_steps_eff,
                rmse_prev,
                corr_prev,
                lag_steps_eff,
                rmse_next,
                corr_next,
            )

            ax.plot(x, input_mean.values, color="tab:blue", linewidth=2.0, label="input(t)")
            ax.plot(x, target_mean.values, color="tab:orange", linewidth=2.0, label="target(t)")
            ax.plot(x, target_prev.values, color="tab:green", linewidth=1.5, linestyle="--", label=f"target(t-{lag_steps_eff})")
            ax.plot(x, target_next.values, color="tab:red", linewidth=1.5, linestyle="--", label=f"target(t+{lag_steps_eff})")
            ax.set_xlabel("Time")
            ax.set_ylabel(_format_y_label(var, unit=_get_var_unit(left_ds, var)))
            ax.grid(True, alpha=0.3)
            ax.set_title(
                " | ".join([
                    f"lag0 rmse={rmse_0:.3f}, corr={corr_0:.3f}",
                    f"t-{lag_steps_eff} rmse={rmse_prev:.3f}, corr={corr_prev:.3f}",
                    f"t+{lag_steps_eff} rmse={rmse_next:.3f}, corr={corr_next:.3f}",
                ]),
                fontsize=10,
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(stage_plot_folder.joinpath(f"{stage}_{var}_lag_diagnostic_timeseries.png"), dpi=200)
        except Exception as exc:
            logger.warning(
                "Failed to generate %s lag diagnostic for %s/%s/%s: %s",
                stage_kind,
                data_type,
                stage,
                var,
                repr(exc),
            )
        finally:
            plt.close(fig)


def plot_stage_climatology_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
) -> None:
    clim_plot_specs = [{**spec, "ds": _build_climatology_ds(spec["ds"])} for spec in plot_specs]
    plot_stage_timeseries(
        logger=logger,
        plots_folder_path=plots_folder_path,
        plot_specs=clim_plot_specs,
        data_type=data_type,
        stage=stage,
        stage_kind=f"{stage_kind} climatology",
        x_dim="month",
        x_label="Month",
        filename_suffix="climatology_timeseries",
        y_label_mode="unit",
    )


def plot_stage_minus_climatology_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
) -> None:
    anom_plot_specs = [{**spec, "ds": _build_ds_minus_climatology_ds(spec["ds"])} for spec in plot_specs]
    plot_stage_timeseries(
        logger=logger,
        plots_folder_path=plots_folder_path,
        plot_specs=anom_plot_specs,
        data_type=data_type,
        stage=stage,
        stage_kind=f"{stage_kind} minus climatology",
        filename_suffix="anomaly_timeseries",
        y_label_mode="unit",
    )


def plot_stage_variance_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
    window: int | None = None,
) -> None:
    variance_plot_specs = [
        {
            **spec,
            "ds": _build_anomaly_variance_ds(spec["ds"], window=window),
        }
        for spec in plot_specs
    ]
    plot_stage_timeseries(
        logger=logger,
        plots_folder_path=plots_folder_path,
        plot_specs=variance_plot_specs,
        data_type=data_type,
        stage=stage,
        stage_kind=f"{stage_kind} anomaly variance",
        filename_suffix="anomaly_variance_timeseries",
        y_label_mode="variance",
    )


def plot_stage_autocorr_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
    nlags: int | None = None,
) -> None:
    autocorr_plot_specs = [
        {
            **spec,
            "ds": _build_anomaly_autocorr_ds(spec["ds"], nlags=nlags),
        }
        for spec in plot_specs
    ]
    plot_stage_timeseries(
        logger=logger,
        plots_folder_path=plots_folder_path,
        plot_specs=autocorr_plot_specs,
        data_type=data_type,
        stage=stage,
        stage_kind=f"{stage_kind} anomaly autocorr",
        x_dim="lag",
        x_label="Lag",
        filename_suffix="anomaly_autocorr_timeseries",
        y_label_mode="autocorr",
    )


def plot_stage_power_spectrum_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
) -> None:
    period_unit = _infer_period_unit(plot_specs[0]["ds"])
    power_plot_specs = [
        {
            **spec,
            "ds": _build_anomaly_power_spectrum_ds(spec["ds"]),
        }
        for spec in plot_specs
    ]
    plot_stage_timeseries(
        logger=logger,
        plots_folder_path=plots_folder_path,
        plot_specs=power_plot_specs,
        data_type=data_type,
        stage=stage,
        stage_kind=f"{stage_kind} anomaly power spectrum",
        x_dim="frequency",
        x_label=f"Period [{period_unit}]",
        filename_suffix="anomaly_power_spectrum_timeseries",
        y_label_mode="power_spectrum",
    )


def plot_stage_temporal_mean_maps(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
) -> None:
    stage_plot_folder = _get_stage_plot_folder(plots_folder_path, data_type)
    logger.info(
        "Generate %s temporal mean maps for %s (%s) in %s",
        stage_kind,
        data_type,
        stage,
        stage_plot_folder,
    )

    common_vars = [
        var for var in _data_vars_for_plotting(plot_specs[0]["ds"])
        if all(var in spec["ds"].data_vars or len(_data_vars_for_plotting(spec["ds"])) == 1 for spec in plot_specs[1:])
    ]
    if not common_vars:
        logger.debug("Skip %s temporal mean maps for %s: no common variables.", stage_kind, data_type)
        return

    for var in common_vars:
        for spec in plot_specs:
            try:
                ds_var = _select_plot_var_flexible(spec["ds"], var)
                unit = _get_var_unit(ds_var, var)
                plot_temporal_mean_map(
                    ds_var,
                    save_path=stage_plot_folder.joinpath(f"{stage}_{var}_{spec['label']}_temporal_mean_map.png"),
                    cbar_label=_format_map_cbar_label(var, unit),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to generate %s temporal mean map for %s/%s/%s: %s",
                    stage_kind,
                    data_type,
                    stage,
                    f"{var}_{spec['label']}",
                    repr(exc),
                )


def plot_stage_metric_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    truth_ds: xr.Dataset,
    data_type: str,
    stage: str,
    stage_kind: str,
) -> None:
    stage_plot_folder = _get_stage_plot_folder(plots_folder_path, data_type)
    time_dim = truth_ds.earthml.guessed_dims.time
    lat_dim = truth_ds.earthml.guessed_dims.latitude
    lon_dim = truth_ds.earthml.guessed_dims.longitude

    if time_dim is None or time_dim not in truth_ds.dims or lat_dim is None or lon_dim is None:
        logger.debug("Skip %s metric timeseries for %s: missing time/lat/lon dimensions.", stage_kind, data_type)
        return

    model_specs = _metric_model_specs(plot_specs, truth_ds)
    if not model_specs:
        logger.debug("Skip %s metric timeseries for %s: no model datasets.", stage_kind, data_type)
        return

    metric_datasets = _build_metric_datasets(
        truth_ds=truth_ds,
        model_specs=model_specs,
        reduce_dims=(lat_dim, lon_dim),
    )
    common_vars = [
        var for var in _data_vars_for_plotting(truth_ds)
        if any(var in spec["ds"].data_vars or len(_data_vars_for_plotting(spec["ds"])) == 1 for spec in model_specs)
    ]
    if not common_vars:
        return

    for metric_name, metric_ds in metric_datasets.items():
        for var in common_vars:
            fig, ax = plt.subplots(figsize=(10, 4))
            try:
                plotted = False
                metric_unit = _metric_unit(metric_name, _get_var_unit(truth_ds, var))
                y_label = _format_metric_label(metric_name, unit=metric_unit)
                for spec in model_specs:
                    model_metric_ds = _select_metric_model_ds(metric_ds, spec["label"])
                    if var not in model_metric_ds.data_vars and len(_data_vars_for_plotting(model_metric_ds)) != 1:
                        continue
                    plotted = True
                    _plot_single_timeseries(
                        logger=logger,
                        ax=ax,
                        ds=model_metric_ds,
                        var=var,
                        time_dim=time_dim,
                        stage_kind=f"{stage_kind} {metric_name}",
                        data_type=data_type,
                        label=f"{spec['label']} {metric_name}",
                        mean_label=f"{spec['mean_label']} {metric_name}",
                        color=spec["color"],
                        y_label=y_label,
                    )
                if not plotted:
                    continue
                ax.legend()
                fig.tight_layout()
                fig.savefig(stage_plot_folder.joinpath(f"{stage}_{var}_{metric_name}_timeseries.png"), dpi=200)
            except Exception as exc:
                logger.warning(
                    "Failed to generate %s metric timeseries for %s/%s/%s/%s: %s",
                    stage_kind,
                    data_type,
                    stage,
                    var,
                    metric_name,
                    repr(exc),
                )
            finally:
                plt.close(fig)


def plot_stage_metric_maps(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    truth_ds: xr.Dataset,
    data_type: str,
    stage: str,
    stage_kind: str,
) -> None:
    stage_plot_folder = _get_stage_plot_folder(plots_folder_path, data_type)
    time_dim = truth_ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in truth_ds.dims:
        logger.debug("Skip %s metric maps for %s: missing time dimension.", stage_kind, data_type)
        return

    model_specs = _metric_model_specs(plot_specs, truth_ds)
    if not model_specs:
        logger.debug("Skip %s metric maps for %s: no model datasets.", stage_kind, data_type)
        return

    metric_datasets = _build_metric_datasets(
        truth_ds=truth_ds,
        model_specs=model_specs,
        reduce_dims=(time_dim,),
    )
    common_vars = [
        var for var in _data_vars_for_plotting(truth_ds)
        if any(var in spec["ds"].data_vars or len(_data_vars_for_plotting(spec["ds"])) == 1 for spec in model_specs)
    ]
    if not common_vars:
        return

    for metric_name, metric_ds in metric_datasets.items():
        for var in common_vars:
            for spec in model_specs:
                try:
                    model_metric_ds = _select_metric_model_ds(metric_ds, spec["label"])
                    if var not in model_metric_ds.data_vars and len(_data_vars_for_plotting(model_metric_ds)) != 1:
                        continue
                    metric_unit = _metric_unit(metric_name, _get_var_unit(truth_ds, var))
                    plot_temporal_mean_map(
                        _select_plot_var_flexible(model_metric_ds, var),
                        save_path=stage_plot_folder.joinpath(f"{stage}_{var}_{spec['label']}_{metric_name}_map.png"),
                        cbar_label=_format_metric_label(metric_name, unit=metric_unit),
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to generate %s metric map for %s/%s/%s/%s/%s: %s",
                        stage_kind,
                        data_type,
                        stage,
                        var,
                        spec["label"],
                        metric_name,
                        repr(exc),
                    )


def run_stage_plot_bundle(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    data_type: str,
    stage: str,
    stage_kind: str,
    residual_label: str,
    residual_mean_label: str,
    lag_steps: int | None = None,
) -> None:
    shared_kwargs = dict(
        logger=logger,
        plots_folder_path=plots_folder_path,
        data_type=data_type,
        stage=stage,
        stage_kind=stage_kind,
    )

    plot_stage_timeseries(
        plot_specs=plot_specs,
        filename_suffix="raw_timeseries",
        **shared_kwargs,
    )
    plot_stage_minus_climatology_timeseries(
        plot_specs=plot_specs,
        **shared_kwargs,
    )
    plot_stage_temporal_mean_maps(
        plot_specs=plot_specs,
        **shared_kwargs,
    )
    plot_stage_residual_timeseries(
        left_ds=left_ds,
        right_ds=right_ds,
        residual_label=residual_label,
        residual_mean_label=residual_mean_label,
        **shared_kwargs,
    )
    plot_stage_lag_diagnostic(
        left_ds=left_ds,
        right_ds=right_ds,
        lag_steps=lag_steps,
        **shared_kwargs,
    )
