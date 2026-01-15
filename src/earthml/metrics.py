from typing import List, Any, Literal, Callable, Optional
from pathlib import Path

from rich import print
from rich.pretty import pprint
from rich.console import Console

import numpy as np
import pandas as pd
from scipy.signal import get_window
import cf_xarray
import xarray as xr
import xskillscore as xs
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os

from .utils import guess_time_dim, guess_lon_dim, guess_lat_dim

MetricFn = Callable[[xr.Dataset, xr.Dataset], xr.Dataset]
FinalFn  = Callable[[xr.Dataset], xr.Dataset]

# Standalone helper methods
def metrics_to_df (metrics_dict, metric_name, kind="scalar"):
    """
    Returns a tidy dataframe with columns:
    period, model, variable, leadtime, value
    Works for scalar metrics that return xarray.Dataset with a leadtime coord.
    """
    rows = []

    for period, res in metrics_dict.items():
        models = res["models"]
        for model_name, model_res in models.items():
            ds = model_res[kind].get(metric_name, None)
            if ds is None:
                continue

            for var in ds.data_vars:
                da = ds[var]

                # drop/squeeze nuisance dims if present
                for dim in list(da.dims):
                    if dim not in ("leadtime",):  # keep leadtime; everything else squeeze if possible
                        if da.sizes.get(dim, 1) == 1:
                            da = da.squeeze(dim, drop=True)

                # if still multi-dim beyond leadtime, flatten remaining dims into rows
                if "leadtime" in da.dims:
                    # iterate leadtime values
                    for lt in da["leadtime"].values:
                        val = da.sel(leadtime=lt).values
                        # if val still array-like, flatten
                        val = np.array(val).reshape(-1)
                        for v in val:
                            rows.append({
                                "period": period,
                                "model": model_name,
                                "metric": metric_name,
                                "variable": var,
                                "leadtime": int(lt),
                                "value": float(v) if np.isfinite(v) else np.nan,
                            })
                else:
                    # no leadtime: single number
                    val = float(da.values) if np.isfinite(da.values) else np.nan
                    rows.append({
                        "period": period,
                        "model": model_name,
                        "metric": metric_name,
                        "variable": var,
                        "leadtime": np.nan,
                        "value": val,
                    })

    df = pd.DataFrame(rows)
    return df

def _extent_from_da (da: xr.DataArray, lon_name="lon", lat_name="lat", pad_deg=2.0):
    lon = da[lon_name].values
    lat = da[lat_name].values

    # reduce to 1D arrays if needed
    lon = np.asarray(lon).ravel()
    lat = np.asarray(lat).ravel()

    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]

    # handle 0..360 longitudes -> convert to -180..180 for plotting
    if lon.size and lon.min() >= 0 and lon.max() > 180:
        lon = ((lon + 180) % 360) - 180

    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())

    # padding
    lon_min -= pad_deg
    lon_max += pad_deg
    lat_min -= pad_deg
    lat_max += pad_deg

    return (lon_min, lon_max, lat_min, lat_max)

def plot_map_metric_grid (
    metrics: dict,
    *,
    metric: str = "nrmse_map",
    kind: str = "map",
    variable: str = "t2m_mse",
    periods: list[str] | None = None,
    leadtimes: list[int] | None = None,
    # model vs diff
    model: str | None = "ECMWF fc",
    diff: bool = False,
    model_a: str = "ECMWF fc",
    model_b: str = "MLFC pr",
    # plotting
    projection=ccrs.PlateCarree(), # Robinson(),
    data_crs=ccrs.PlateCarree(),
    add_coastlines: bool = True,
    figsize_per_cell=(4.2, 2.8),
    cmap: str | None = None,
    robust: bool = True,
    q: float = 0.02,
    symmetric_diff: bool = True,
    auto_extent=True,
    pad_deg=0,
    save_path: str | None = None,
):
    """
    rows = periods, cols = leadtimes
    Each panel is a map for `variable` from metrics[period]['models'][...][kind][metric].
    If diff=True, plots model_b - model_a.
    """

    # --- choose periods ---
    if periods is None:
        periods = list(metrics.keys())
    if not periods:
        raise ValueError("No periods found in metrics.")

    # --- helpers ---
    def _get_ds(tp: str, mname: str) -> xr.Dataset:
        ds = metrics[tp]["models"][mname][kind][metric]
        return ds.squeeze(drop=True)

    def _panel_da(tp: str) -> xr.DataArray:
        if diff:
            ds_a = _get_ds(tp, model_a)
            ds_b = _get_ds(tp, model_b)
            # FIX: xarray aligns via function, not method
            ds_a, ds_b = xr.align(ds_a, ds_b, join="inner")
            da = ds_b[variable] - ds_a[variable]
        else:
            ds = _get_ds(tp, model)
            da = ds[variable]

        # squeeze any remaining singleton dims (common: realization=1, number=1, etc.)
        return da.squeeze(drop=True)

    # --- find reference for leadtimes ---
    ref_da = None
    for tp in periods:
        try:
            da = _panel_da(tp)
            ref_da = da
            break
        except Exception:
            continue
    if ref_da is None:
        raise ValueError(f"Could not find data for metric='{metric}', variable='{variable}'.")

    if "leadtime" in ref_da.coords:
        if leadtimes is None:
            leadtimes = [int(x) for x in ref_da["leadtime"].values]
        else:
            leadtimes = [int(x) for x in leadtimes]
    else:
        leadtimes = [None]  # single column

    nrows = len(periods)
    ncols = len(leadtimes)

    # --- global vmin/vmax across all panels for comparability ---
    vals = []
    for tp in periods:
        try:
            da = _panel_da(tp)
            if "leadtime" in da.coords and leadtimes[0] is not None:
                for lt in leadtimes:
                    if lt in da["leadtime"].values:
                        vals.append(da.sel(leadtime=lt))
            else:
                vals.append(da)
        except Exception:
            continue

    if not vals:
        raise ValueError("No data available to plot after filtering periods/leadtimes.")

    all_data = np.concatenate([np.ravel(v.values) for v in vals])
    all_data = all_data[np.isfinite(all_data)]
    if all_data.size == 0:
        raise ValueError("All selected data are NaN/inf; nothing to plot.")

    if robust:
        vmin = float(np.quantile(all_data, q))
        vmax = float(np.quantile(all_data, 1 - q))
    else:
        vmin = float(np.min(all_data))
        vmax = float(np.max(all_data))

    if diff and symmetric_diff:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    # --- figure & axes ---
    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": projection},
        squeeze=False,
    )

    mappable = None

    for r, tp in enumerate(periods):
        for c, lt in enumerate(leadtimes):
            ax = axes[r, c]

            if add_coastlines:
                ax.coastlines(linewidth=0.6)

            try:
                da = _panel_da(tp)

                # select leadtime if present
                if lt is not None and "leadtime" in da.coords:
                    if lt not in da["leadtime"].values:
                        ax.set_title(f"{tp}\nLT={lt}h (missing)")
                        ax.axis("off")
                        continue
                    da2 = da.sel(leadtime=lt)
                    lt_title = f"LT={lt}h"
                else:
                    da2 = da
                    lt_title = ""

                if auto_extent:
                    lon_name, lat_name = guess_lon_dim(da2), guess_lat_dim(da2)
                    ext = _extent_from_da(da2, lon_name=lon_name, lat_name=lat_name, pad_deg=pad_deg)
                    ax.set_extent(ext, crs=data_crs)
                else:
                    ax.set_global()

                # plot
                im = da2.plot(
                    ax=ax,
                    transform=data_crs,
                    add_colorbar=False,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                mappable = im

                if diff:
                    ax.set_title(f"{tp}\n{lt_title}\nΔ({model_b}−{model_a})")
                else:
                    ax.set_title(f"{tp}\n{lt_title}\n{model}")

            except Exception as e:
                ax.set_title(f"{tp}\nerror")
                ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.axis("off")

    # shared colorbar
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
        if diff:
            cbar.set_label(f"Δ {metric} ({model_b} − {model_a})")
        else:
            cbar.set_label(metric)

    # title
    if diff:
        fig.suptitle(f"Δ {metric} maps ({model_b} − {model_a}) — {variable}", y=1.02)
    else:
        fig.suptitle(f"{metric} maps — {variable} — {model}", y=1.02)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

    return fig

class Metrics:
    def __init__ (self, truth: xr.Dataset, data: xr.Dataset | List[xr.Dataset], truth_name: str , data_name: str | List[str]):
        self.truth = truth
        self.data = data if isinstance(data, list) else [data]

        self.truth_name = truth_name
        self.data_name = data_name if isinstance(data_name, list) else [data_name]

        self.time_dim, self.lat_dim, self.lon_dim = self._get_and_rename_dim()

        # drop _has_var
        truth_sel_vars = [v for v in self.truth.data_vars if v != '_has_var']
        self.truth = self.truth[truth_sel_vars]
        sel_data = []
        for d in self.data:
            data_sel_vars = [v for v in d.data_vars if v != '_has_var']
            sel_data.append(d[data_sel_vars])
        self.data = sel_data

    def _get_and_rename_dim (self):
        # Rename all data vars to match truth vars
        # truth_vars = set(self.truth)
        # for i, d in enumerate(self.data):
        #     d_vars = list(d.data_vars)
        #     if set(d_vars) != set(truth_vars):
        #         print(f"Renaming {self.data_name[i]} vars to match {self.truth_name}")
        #         rename_map = {old: new for old, new in zip(d_vars, truth_vars)}
        #         self.data[i] = d.rename_vars(rename_map)
        # Rename time, lat and lon data dims to match truth dims
        dims = (guess_time_dim(self.truth), guess_lat_dim(self.truth), guess_lon_dim(self.truth))
        for i, d in enumerate(self.data):
            data_dims = (guess_time_dim(d), guess_lat_dim(d), guess_lon_dim(d))
            for dim, data_dim in zip(dims, data_dims):
                if data_dim != dim:
                    self.data[i] = d.rename_dims({data_dim: dim})
        return dims # time, lat, lon

    def _geo_avg (self, data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """Geographically weighted mean over lat/lon using cos(lat)."""
        # weights must be an xarray DataArray aligned to the latitude dimension
        w = xr.DataArray(
            np.cos(np.deg2rad(data[self.lat_dim])),
            coords={self.lat_dim: data[self.lat_dim]},
            dims=(self.lat_dim,),
        )

        # weighted mean over spatial dims; keep time (and any other non-spatial dims)
        return data.weighted(w).mean(dim=(self.lat_dim, self.lon_dim), skipna=True)

    # Metrics
    def _generic_metric (
        self,
        parent_metric_fn: MetricFn,
        final_metric_fn: Optional[FinalFn] = None,
        order: Literal["3d", "2d", "1d"] = "1d",
        geo_weighted: bool = True,
    ) -> list[xr.Dataset]:

        out: list[xr.Dataset] = []

        for d in self.data:
            metric = parent_metric_fn(self.truth, d)

            if order == "1d":
                if geo_weighted:
                    val = self._geo_avg(metric).mean(dim=self.time_dim, skipna=True)
                else:
                    val = metric.mean(dim=(self.time_dim, self.lat_dim, self.lon_dim), skipna=True)
            elif order == "2d":
                val = metric.mean(dim=self.time_dim, skipna=True)
            elif order == "3d":
                val = metric
            else:
                raise ValueError(f"Invalid order: {order}")

            if final_metric_fn is not None:
                val = final_metric_fn(val)

            out.append(val)

        return out

    def mae (self, order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True) -> List[xr.Dataset]:
        return self._generic_metric(lambda truth, pred: abs(truth - pred), order=order, geo_weighted=geo_weighted)

    @staticmethod
    def _err_field (truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
        return truth - pred

    def err (self, order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True) -> List[xr.Dataset]:
        # 2d and 1d cases are bias
        return self._generic_metric(self._err_field, order=order, geo_weighted=geo_weighted)

    def std_of_errs (self, order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True) -> list[xr.Dataset]:
        """
        Standard deviation of errors
        """
        return self._generic_metric(
            parent_metric_fn=self._err_field,
            final_metric_fn=lambda x: x.std(skipna=True),
            order=order,
            geo_weighted=geo_weighted,
        )
    
    def stderr (self,order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True) -> list[xr.Dataset]:
        """
        Standard error of the mean
        """
        def final_fn(x: xr.Dataset) -> xr.Dataset:
            n = x.count()
            return x.std(skipna=True) / np.sqrt(n)

        return self._generic_metric(
            parent_metric_fn=self._err_field,
            final_metric_fn=final_fn,
            order=order,
            geo_weighted=geo_weighted,
        )

    def rmse (self, order: Literal["2d", "1d"] = "1d", geo_weighted: bool = True) -> List[xr.Dataset]:
        return self._generic_metric(lambda truth, pred: (truth - pred)**2, final_metric_fn=np.sqrt, order=order, geo_weighted=geo_weighted)
    
    def nrmse_map (self, order: Literal["2d", "1d"] = "2d", geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        NRMSE normalized by truth std over time (map-wise).

        - order="2d": returns a 2D NRMSE map: RMSE_map / std_map
        - order="1d": returns a scalar per dataset by spatially averaging the 2D NRMSE map
                    (geo-weighted if geo_weighted=True)

        Notes:
        - std_map = std(truth over time_dim)
        - cells where std_map <= eps are set to NaN (ignored by skipna reductions)
        """

        # RMSE map per dataset (time-reduced)
        rmse_maps = self.rmse(order="2d", geo_weighted=geo_weighted) # geo_weighted irrelevant for 2d

        # Truth std map over time (same for all predictions)
        std_map = self.truth.std(dim=self.time_dim, skipna=True)
        denom = xr.where(std_map > eps, std_map, np.nan)

        out: list[xr.Dataset] = []
        for rmse_map in rmse_maps:
            nrmse_map = rmse_map / denom

            if order == "2d":
                out.append(nrmse_map)
            elif order == "1d":
                if geo_weighted:
                    out.append(self._geo_avg(nrmse_map)) # already a scalar Dataset (since nrmse_map has no time dim)
                else:
                    out.append(nrmse_map.mean(dim=(self.lat_dim, self.lon_dim), skipna=True))
            else:
                raise ValueError(f"Invalid order: {order}")

        return out

    def nrmse_global (self, geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        Global NRMSE normalized by global truth standard deviation.

        Returns one scalar per dataset.

        Definition:
        NRMSE = RMSE_global / std_global(truth)

        If geo_weighted=True:
        - spatial means are cosine(lat) weighted
        - time is unweighted
        """

        # Denominator: global std of truth
        if geo_weighted:
            # spatially averaged time series, then std over time
            truth_bar = self._geo_avg(self.truth)        # dims: time
            denom = truth_bar.std(dim=self.time_dim, skipna=True)
        else:
            denom = self.truth.std(
                dim=(self.time_dim, self.lat_dim, self.lon_dim),
                skipna=True,
            )

        denom = xr.where(denom > eps, denom, np.nan)

        out: list[xr.Dataset] = []

        for d in self.data:
            # Numerator: global RMSE
            if geo_weighted:
                err2_bar = self._geo_avg((self.truth - d) ** 2)   # dims: time
                mse = err2_bar.mean(dim=self.time_dim, skipna=True)
            else:
                mse = ((self.truth - d) ** 2).mean(
                    dim=(self.time_dim, self.lat_dim, self.lon_dim),
                    skipna=True,
                )

            rmse = mse ** 0.5
            out.append(rmse / denom)

        return out

    def r2_map (self) -> list[xr.Dataset]:
        # SST: sum over time of squared anomalies (2D field per var)
        anom = self.truth - self.truth.mean(dim=self.time_dim, skipna=True)
        sst = (anom ** 2).sum(dim=self.time_dim, skipna=True)

        out: list[xr.Dataset] = []
        for d in self.data:
            sse = ((self.truth - d) ** 2).sum(dim=self.time_dim, skipna=True)
            r2 = 1.0 - sse / xr.where(sst > 0, sst, np.nan)
            out.append(r2)

        return out

    def r2_global (self, geo_weighted: bool = True) -> list[xr.Dataset]:
        # anomalies over time
        anom = self.truth - self.truth.mean(dim=self.time_dim, skipna=True)
        sst_t = (anom ** 2).sum(dim=self.time_dim, skipna=True)  # 2D field

        out: list[xr.Dataset] = []
        for d in self.data:
            sse_t = ((self.truth - d) ** 2).sum(dim=self.time_dim, skipna=True)  # 2D field

            if geo_weighted:
                sse = self._geo_avg(sse_t)   # scalar or 1D depending on _geo_avg
                sst = self._geo_avg(sst_t)
            else:
                sse = sse_t.mean(dim=(self.lat_dim, self.lon_dim), skipna=True)
                sst = sst_t.mean(dim=(self.lat_dim, self.lon_dim), skipna=True)

            r2 = 1 - sse / xr.where(sst > 0, sst, np.nan)
            out.append(r2)

        return out

    def mape (self, order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        Mean Absolute Percentage Error.

        Returns:
        - order="3d": pointwise APE (|err|/|truth| * 100)
        - order="2d": time-mean of APE (2D field)
        - order="1d": time-mean then spatial mean (scalar), geo-weighted if requested
        """
        def _mape_field (truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
            denom = xr.where(np.abs(truth) > eps, np.abs(truth), np.nan)
            return (np.abs(truth - pred) / denom) * 100.0

        return self._generic_metric(_mape_field, order=order, geo_weighted=geo_weighted)

    def smape (self, order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True) -> list[xr.Dataset]:
        """
        Symmetric Mean Absolute Percentage Error
        """
        def _smape_field (truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
            denom = np.abs(truth) + np.abs(pred)
            denom = xr.where(denom > 0, denom, np.nan)
            return 100.0 * (2 * np.abs(truth - pred) / denom)

        return self._generic_metric(_smape_field, order=order, geo_weighted=geo_weighted)

    def compute_all_metrics(self, *, geo_weighted: bool = True, eps: float = 0.0) -> dict[str, Any]:
        """
        Compute a standard suite of metrics and return:
          res["models"][model]["scalar"][metric] -> xr.Dataset (0D per var)
          res["models"][model]["map"][metric]    -> xr.Dataset (2D per var: lat/lon)

        Notes
        -----
        - Scalar metrics use geo-weighting if geo_weighted=True (where applicable).
        - 3D fields are intentionally not returned.
        """
        def by_model(lst: list[xr.Dataset]) -> dict[str, xr.Dataset]:
            return {name: lst[i] for i, name in enumerate(self.data_name)}

        # metric name -> callables producing list[xr.Dataset] aligned with self.data
        scalar_fns = {
            "mae":           lambda: self.mae(order="1d", geo_weighted=geo_weighted),
            "bias":          lambda: self.err(order="1d", geo_weighted=geo_weighted),
            "rmse":          lambda: self.rmse(order="1d", geo_weighted=geo_weighted),
            "stderr":        lambda: self.stderr(order="1d", geo_weighted=geo_weighted),
            "std_of_errs":   lambda: self.std_of_errs(order="1d", geo_weighted=geo_weighted),
            "mape":          lambda: self.mape(order="1d", geo_weighted=geo_weighted, eps=eps),
            "smape":         lambda: self.smape(order="1d", geo_weighted=geo_weighted),
            "nrmse_global":  lambda: self.nrmse_global(geo_weighted=geo_weighted, eps=eps),
            "nrmse_map_mean":lambda: self.nrmse_map(order="1d", geo_weighted=geo_weighted, eps=eps),
            "r2_global":     lambda: self.r2_global(geo_weighted=geo_weighted),
        }

        map_fns = {
            "mae":          lambda: self.mae(order="2d", geo_weighted=geo_weighted),
            "bias":         lambda: self.err(order="2d", geo_weighted=geo_weighted),
            "rmse":         lambda: self.rmse(order="2d", geo_weighted=geo_weighted),
            "stderr":       lambda: self.stderr(order="2d", geo_weighted=geo_weighted),
            "std_of_errs":  lambda: self.std_of_errs(order="2d", geo_weighted=geo_weighted),
            "mape":         lambda: self.mape(order="2d", geo_weighted=geo_weighted, eps=eps),
            "smape":        lambda: self.smape(order="2d", geo_weighted=geo_weighted),
            "nrmse_map":    lambda: self.nrmse_map(order="2d", geo_weighted=geo_weighted, eps=eps),
            "r2_map":       lambda: self.r2_map(),
        }

        # init models
        models: dict[str, dict[str, dict[str, xr.Dataset]]] = {
            name: {"scalar": {}, "map": {}} for name in self.data_name
        }

        # fill scalars
        for metric, fn in scalar_fns.items():
            for model, ds in by_model(fn()).items():
                models[model]["scalar"][metric] = ds

        # fill maps
        for metric, fn in map_fns.items():
            for model, ds in by_model(fn()).items():
                models[model]["map"][metric] = ds

        return {
            "meta": {
                "truth_name": self.truth_name,
                "data_names": list(self.data_name),
                "dims": {"time": self.time_dim, "lat": self.lat_dim, "lon": self.lon_dim},
                "geo_weighted": geo_weighted,
                "eps": eps,
            },
            "models": models,
        }

    # @staticmethod
    # def fss (a, b, dims: Tuple, threshold, eps):
    #     """Fraction Skill Score"""
    #     A, B = (a > threshold).astype(float), (b > threshold).astype(float)
    #     return 1 - ((A - B)**2).mean(dim=time_dim) / ((A**2 + B**2).mean(dim=time_dim) + eps)

    # @staticmethod
    # def sal (a, b, dims: Tuple, eps): #TODO better understand this metric
    #     """Std ratio, Avg (mean) ratio, weighted mean difference (L)"""
    #     return {
    #         "S": (b.std(dim=time_dim) - a.std(dim=time_dim)) / (a.std(dim=time_dim) + eps),
    #         "A": (b.mean(dim=time_dim) - a.mean(dim=time_dim)) / (a.mean(dim=time_dim) + eps),
    #         "L": (b.weighted(abs(b)).mean(dim=time_dim) - a.weighted(abs(a)).mean(dim=time_dim)) # TODO probably wrong, check weights
    #     }
    @staticmethod
    def spectral_metrics (a, b, time_dim, spatial_dims, eps):
        import numpy as np

        # Helper: FFT power with consistent axis order
        def fft_power(da, time_dim, spatial_dims):
            arr = da.transpose(time_dim, *spatial_dims).astype("float64").values
            fft = np.fft.rfftn(arr, axes=(-2, -1)) # 2D spatial FFT
            return fft

        # Get FFTs
        A = fft_power(a, time_dim, spatial_dims)
        B = fft_power(b, time_dim, spatial_dims)

        # Power spectra (averaged over time axis = 0)
        Pa = np.mean(np.abs(A) ** 2, axis=0)
        Pb = np.mean(np.abs(B) ** 2, axis=0)

        # Power ratio
        ratio = Pb / (Pa + eps)

        # Cross-spectrum
        cross = np.mean(A * np.conj(B), axis=0)

        # Coherence: |E[AB*]| / sqrt(E[|A|^2] E[|B|^2])
        # IMPORTANT: avoid overflow by not doing Pa*Pb directly
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            denom = np.sqrt(Pa) * np.sqrt(Pb) + eps
            coh = np.abs(cross) / denom
            # Optional: coherence should be in [0, 1]
            coh = np.clip(coh, 0, 1)

        return {
            "power_a": Pa,
            "power_b": Pb,
            "ratio": ratio,
            "coherence": coh
        }

class PowerSpectrum:
    def __init__(
        self,
        data: xr.Dataset,
        dx_deg: float | None = None,
        dy_deg: float | None = None,
        n_bins: int = 40,
    ):
        """
        Compute time-mean radial power spectra (as function of wavelength in km)
        for all variables in an xarray Dataset on a lat-lon grid.

        Parameters
        ----------
        data : xr.Dataset
            Input dataset with at least lat/lon coordinates.
        dx_deg, dy_deg : float or None
            Horizontal resolution in degrees. If None, inferred from coordinates.
        n_bins : int
            Number of radial bins in wavelength space.
        """
        self.data = data
        self.lat_name = self.data.cf['latitude'].name
        self.lon_name = self.data.cf['longitude'].name
        self.time_name = self.data.cf['time'].name
        self.n_bins = n_bins

        # --- Infer grid information (lat array, dx_deg, dy_deg) ---
        lat_1d, lon_1d, dx_deg, dy_deg = self._infer_grid_info(
            self.lat_name, self.lon_name, dx_deg, dy_deg
        )
        self.lat_1d = lat_1d
        self.lon_1d = lon_1d
        self.dx_deg = dx_deg
        self.dy_deg = dy_deg

        ps_vars = {}

        for v in self.data.data_vars:
            da = self.data[v]

            # Ensure it has the spatial dims
            if not (self.lat_name in da.dims and self.lon_name in da.dims):
                # Skip non-spatial variables
                continue

            # --- Time-average of the power spectrum ---
            if self.time_name in da.dims:
                time_dim = self.time_name
                nt = da.sizes[time_dim]

                lam_centers = None
                spectra = []

                for it in range(nt):
                    slice_da = da.isel({time_dim: it})
                    field2d = self._prepare_field2d(slice_da)

                    lam, ps = self.compute_radial_spectrum(
                        field2d,
                        self.lat_1d,
                        self.dx_deg,
                        self.dy_deg,
                        n_bins=self.n_bins,
                    )

                    if lam_centers is None:
                        lam_centers = lam
                    else:
                        if not np.allclose(lam_centers, lam):
                            raise ValueError(f"Inconsistent wavelength grid for variable {v}")

                    spectra.append(ps)

                spectra = np.stack(spectra, axis=0)  # (time, wavelength)
                ps_mean = np.nanmean(spectra, axis=0)  # time-mean spectrum
            else:
                # No time dimension: just one spectrum
                field2d = self._prepare_field2d(da)
                lam_centers, ps_mean = self.compute_radial_spectrum(
                #     field2d,
                #     self.lat_1d,
                #     self.dx_deg,
                #     self.dy_deg,
                #     n_bins=self.n_bins,
                # )
                    field2d,
                    lat_1d,
                    dx_deg,
                    dy_deg,
                    n_bins=40,
                    nperseg_y=1024,
                    nperseg_x=1024,
                    noverlap_y=64,
                    noverlap_x=64,
                    window="hann",
                )


            # Store as DataArray with coordinate "wavelength_km"
            ps_vars[v] = xr.DataArray(
                ps_mean,
                coords={"wavelength_km": lam_centers},
                dims=("wavelength_km",),
                name=v,
            )

        # Dataset of power spectra, with common wavelength axis
        self.ps = xr.Dataset(ps_vars)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _infer_grid_info(
        self,
        lat_name: str,
        lon_name: str,
        dx_deg: float | None,
        dy_deg: float | None,
    ):
        """Infer 1D lat/lon arrays and degree spacing from the dataset."""
        if lat_name not in self.data.coords or lon_name not in self.data.coords:
            raise ValueError(
                f"Dataset must have '{lat_name}' and '{lon_name}' coordinates."
            )

        lat = self.data[lat_name]
        lon = self.data[lon_name]

        if lat.ndim != 1 or lon.ndim != 1:
            raise NotImplementedError(
                "Currently only 1D lat/lon coordinates are supported."
            )

        lat_1d = lat.values
        lon_1d = lon.values

        # Infer dy_deg and dx_deg if not provided
        if dy_deg is None:
            dy_deg = float(np.nanmean(np.diff(lat_1d)))
        if dx_deg is None:
            # If lon wraps (e.g. 0..360), we still assume uniform spacing
            diffs = np.diff(lon_1d)
            dx_deg = float(np.nanmean(diffs))

        return lat_1d, lon_1d, dx_deg, dy_deg

    def _prepare_field2d(self, da: xr.DataArray) -> np.ndarray:
        """
        Take a DataArray and reduce it to a 2D (lat, lon) field by:
        - averaging over any non-spatial dims
        - transposing to (lat, lon)
        """
        dims_to_avg = [d for d in da.dims if d not in (self.lat_name, self.lon_name)]
        if len(dims_to_avg) > 0:
            da2 = da.mean(dim=dims_to_avg)
        else:
            da2 = da

        # Reorder so that lat is first, lon is second
        da2 = da2.transpose(self.lat_name, self.lon_name)

        field2d = da2.values
        if field2d.ndim != 2:
            raise ValueError(f"Expected 2D field after reduction, got shape {field2d.shape}")
        if field2d.shape[0] != self.lat_1d.size:
            raise ValueError("Latitude size mismatch between field and lat_1d.")
        return field2d

    # ------------------------------------------------------------------
    # Core spectrum computation
    # ------------------------------------------------------------------
    # @staticmethod
    # def compute_radial_spectrum(
    #     field2d: np.ndarray,
    #     lat_1d: np.ndarray,
    #     dx_deg: float,
    #     dy_deg: float,
    #     n_bins: int = 40,
    # ):
    #     """
    #     Compute 1D radial average power spectrum from a 2D field, and express it
    #     as a function of wavelength (km).

    #     - field2d is on a regular lat-lon grid (in degrees).
    #     - FFT is done in index/degree space; then we convert to physical km
    #       using spherical geometry:
    #         1° lat  ≈ 111.32 km
    #         1° lon  ≈ 111.32 * cos(lat) km
    #     - This handles varying dx (km) with latitude.

    #     Parameters
    #     ----------
    #     field2d : 2D ndarray
    #         Field on (lat, lon) grid, shape (ny, nx).
    #     lat_1d : 1D ndarray
    #         Latitude values (degrees), length ny.
    #     dx_deg, dy_deg : float
    #         Grid resolution in degrees (assumed uniform).
    #     n_bins : int
    #         Number of radial bins in wavelength space.

    #     Returns
    #     -------
    #     lam_centers : 1D ndarray
    #         Bin-center wavelengths in km (log-spaced).
    #     radial_power : 1D ndarray
    #         Radially-averaged power for each wavelength bin.
    #     """
    #     field2d = np.asarray(field2d)
    #     if field2d.ndim != 2:
    #         raise ValueError(f"compute_radial_spectrum expects 2D data, got {field2d.shape}")

    #     ny, nx = field2d.shape
    #     if lat_1d.shape[0] != ny:
    #         raise ValueError("lat_1d length must match field2d.shape[0]")

    #     # 2D real FFT
    #     fft = np.fft.rfftn(field2d)
    #     power = np.abs(fft) ** 2
    #     ny_p, nx_r = power.shape
    #     if ny_p != ny:
    #         raise RuntimeError("Internal size mismatch after FFT.")

    #     # Frequencies in cycles per degree
    #     ky_deg = np.fft.fftfreq(ny, d=dy_deg)       # shape (ny,)
    #     kx_deg = np.fft.rfftfreq(nx, d=dx_deg)      # shape (nx_r,)

    #     # Convert to wavenumber in cycles per km (km^-1), accounting for latitude
    #     km_per_deg_lat = 111.32
    #     # latitude-dependent km per deg in longitude
    #     lat_rad = np.deg2rad(lat_1d)
    #     km_per_deg_lon = km_per_deg_lat * np.cos(lat_rad)  # shape (ny,)

    #     # ky: same for every latitude row
    #     ky_km = ky_deg / km_per_deg_lat                 # shape (ny,)
    #     ky_km_2d = ky_km[:, np.newaxis]                 # (ny, 1)

    #     # kx: depends on latitude row via km_per_deg_lon
    #     # kx_deg is (nx_r,), km_per_deg_lon is (ny,)
    #     kx_km_2d = kx_deg[np.newaxis, :] / km_per_deg_lon[:, np.newaxis]

    #     # 2D radial wavenumber magnitude (km^-1)
    #     k_mag = np.sqrt(kx_km_2d**2 + ky_km_2d**2)

    #     # Avoid division by zero at k=0
    #     mask_nonzero = k_mag > 0
    #     if not mask_nonzero.any():
    #         raise RuntimeError("All wavenumbers are zero; something is wrong with the grid.")

    #     # Wavelength in km
    #     wavelength = np.zeros_like(k_mag)
    #     wavelength[mask_nonzero] = 1.0 / k_mag[mask_nonzero]

    #     # Define radial bins in wavelength space (log-spaced)
    #     lam_min = np.nanmin(wavelength[mask_nonzero])
    #     lam_max = np.nanmax(wavelength[mask_nonzero])

    #     lam_bins = np.logspace(np.log10(lam_min), np.log10(lam_max), n_bins)
    #     lam_centers = 0.5 * (lam_bins[:-1] + lam_bins[1:])

    #     radial_power = np.zeros_like(lam_centers)

    #     for i in range(len(lam_centers)):
    #         binmask = (wavelength >= lam_bins[i]) & (wavelength < lam_bins[i + 1])
    #         if binmask.any():
    #             radial_power[i] = power[binmask].mean()
    #         else:
    #             radial_power[i] = np.nan

    #     return lam_centers, radial_power

    @staticmethod
    def compute_radial_spectrum(
        field2d: np.ndarray,
        lat_1d: np.ndarray,
        dx_deg: float,
        dy_deg: float,
        n_bins: int = 40,
        nperseg_y: int | None = None,
        nperseg_x: int | None = None,
        noverlap_y: int | None = None,
        noverlap_x: int | None = None,
        window: str = "hann",
    ):
        """
        Welch-style 2D radial power spectrum expressed as a function of wavelength (km).

        - field2d is on a regular lat–lon grid (in degrees).
        - We split the field into overlapping 2D segments, window each segment,
        take a 2D rFFT, square (power), and average over segments (Welch method).
        - Then we convert to physical wavenumber using spherical geometry:
            1° lat  ≈ 111.32 km
            1° lon  ≈ 111.32 * cos(lat) km
        and finally to wavelength λ = 1 / |k| (km).
        - dx varies with latitude through cos(lat); dy is constant.

        Parameters
        ----------
        field2d : 2D ndarray
            Field on (lat, lon) grid, shape (ny, nx).
        lat_1d : 1D ndarray
            Latitude values (degrees), length ny.
        dx_deg, dy_deg : float
            Grid resolution in degrees (assumed uniform in lat/lon index).
        n_bins : int
            Number of radial bins in wavelength space (edges); centers are n_bins-1.
        nperseg_y, nperseg_x : int or None
            Segment size in y and x. If None, chosen automatically.
        noverlap_y, noverlap_x : int or None
            Overlap between segments. If None, 50% overlap.
        window : str
            Window name passed to scipy.signal.get_window (e.g. 'hann').

        Returns
        -------
        lam_centers : 1D ndarray
            Bin-center wavelengths in km (log-spaced), length n_bins - 1.
        radial_power : 1D ndarray
            Radially-averaged power for each wavelength bin.
        """
        field2d = np.asarray(field2d)
        if field2d.ndim != 2:
            raise ValueError(f"compute_radial_spectrum expects 2D data, got {field2d.shape}")

        ny, nx = field2d.shape
        if lat_1d.shape[0] != ny:
            raise ValueError("lat_1d length must match field2d.shape[0].")

        # --- Choose segment sizes and overlaps (Welch parameters) ---
        if nperseg_y is None:
            nperseg_y = min(256, ny)  # or tune as you like
        if nperseg_x is None:
            nperseg_x = min(256, nx)

        nperseg_y = min(nperseg_y, ny)
        nperseg_x = min(nperseg_x, nx)

        if noverlap_y is None:
            noverlap_y = nperseg_y // 2
        if noverlap_x is None:
            noverlap_x = nperseg_x // 2

        step_y = max(nperseg_y - noverlap_y, 1)
        step_x = max(nperseg_x - noverlap_x, 1)

        y_starts = np.arange(0, ny - nperseg_y + 1, step_y)
        x_starts = np.arange(0, nx - nperseg_x + 1, step_x)

        if len(y_starts) == 0 or len(x_starts) == 0:
            # Fall back to a single segment
            y_starts = np.array([0])
            x_starts = np.array([0])
            nperseg_y = ny
            nperseg_x = nx

        # --- 1D frequency grids for a segment (in cycles per degree) ---
        ky_deg = np.fft.fftfreq(nperseg_y, d=dy_deg)     # (nperseg_y,)
        kx_deg = np.fft.rfftfreq(nperseg_x, d=dx_deg)    # (nperseg_x_r,)

        nseg_x_r = kx_deg.size

        # --- Window (Welch taper) ---
        wy = get_window(window, nperseg_y, fftbins=True)
        wx = get_window(window, nperseg_x, fftbins=True)
        window_2d = wy[:, None] * wx[None, :]  # outer product

        # We'll accumulate power in radial bins
        lam_bins = None
        lam_centers = None
        radial_sum = None
        radial_count = None

        km_per_deg_lat = 111.32  # approximate

        # --- Loop over segments (Welch) ---
        for y0 in y_starts:
            y1 = y0 + nperseg_y
            lat_seg = lat_1d[y0:y1]  # (nperseg_y,)
            lat_rad = np.deg2rad(lat_seg)
            km_per_deg_lon_seg = km_per_deg_lat * np.cos(lat_rad)  # (nperseg_y,)

            # Frequencies in cycles per km for this segment
            ky_km = ky_deg / km_per_deg_lat          # (nperseg_y,)
            ky_km_2d = ky_km[:, None]                # (nperseg_y, 1)

            # kx depends on latitude for each row via km_per_deg_lon_seg
            kx_km_2d = kx_deg[None, :] / km_per_deg_lon_seg[:, None]  # (nperseg_y, nperseg_x_r)

            # 2D radial wavenumber magnitude (km^-1)
            k_mag = np.sqrt(kx_km_2d**2 + ky_km_2d**2)
            mask_nonzero = k_mag > 0

            # Wavelength (km)
            wavelength = np.zeros_like(k_mag)
            wavelength[mask_nonzero] = 1.0 / k_mag[mask_nonzero]

            # Define wavelength bins once (from first segment)
            if lam_bins is None:
                lam_min = np.nanmin(wavelength[mask_nonzero])
                lam_max = np.nanmax(wavelength[mask_nonzero])

                lam_bins = np.logspace(np.log10(lam_min), np.log10(lam_max), n_bins)
                lam_centers = 0.5 * (lam_bins[:-1] + lam_bins[1:])
                radial_sum = np.zeros_like(lam_centers)
                radial_count = np.zeros_like(lam_centers, dtype=np.int64)

            for x0 in x_starts:
                x1 = x0 + nperseg_x

                # Extract segment and apply window
                seg = field2d[y0:y1, x0:x1]
                seg = seg - np.nanmean(seg)  # detrend (remove mean)
                seg_win = seg * window_2d

                # 2D rFFT on segment
                fft_seg = np.fft.rfftn(seg_win)
                power_seg = np.abs(fft_seg) ** 2  # (nperseg_y, nperseg_x_r)

                # Radial binning in wavelength space
                for i in range(lam_centers.size):
                    binmask = (wavelength >= lam_bins[i]) & (wavelength < lam_bins[i + 1])
                    if binmask.any():
                        radial_sum[i] += power_seg[binmask].sum()
                        radial_count[i] += int(binmask.sum())

        # --- Final average over all pixels in all segments ---
        radial_power = np.full_like(radial_sum, np.nan, dtype=float)
        valid = radial_count > 0
        radial_power[valid] = radial_sum[valid] / radial_count[valid]

        return lam_centers, radial_power


    # ------------------------------------------------------------------
    # Optional plotting helper
    # ------------------------------------------------------------------
    def plot_all(
        self,
        normalize: bool = True,
        logx: bool = True,
        logy: bool = True,
        savepath: str | None = None,
        show: bool = True,
    ):
        """
        Quick comparison plot of spectra for all variables in self.ps.

        Parameters
        ----------
        normalize : bool
            If True, divide each spectrum by its max (compare shapes).
        logx, logy : bool
            Use log scale for x and/or y.
        savepath : str or None
            If given, save figure to this path.
        show : bool
            If True, call plt.show().
        """
        import matplotlib.pyplot as plt
        import os

        plt.figure(figsize=(6, 4))

        for v in self.ps.data_vars:
            da = self.ps[v]
            lam = da["wavelength_km"].values
            p = da.values.astype(float)

            mask = np.isfinite(lam) & np.isfinite(p)
            if not mask.any():
                continue

            lam = lam[mask]
            p = p[mask]

            if normalize and p.max() > 0:
                p = p / p.max()

            if logx and logy:
                plt.loglog(lam, p, label=str(v))
            elif logx:
                plt.semilogx(lam, p, label=str(v))
            elif logy:
                plt.semilogy(lam, p, label=str(v))
            else:
                plt.plot(lam, p, label=str(v))

        plt.xlabel("Wavelength (km)")
        plt.ylabel("Power" + (" (normalized)" if normalize else ""))
        plt.legend()
        plt.tight_layout()

        if savepath is not None:
            os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
            plt.savefig(savepath, dpi=150)

        if show:
            plt.show()
        else:
            plt.close()
