#!/usr/bin/env python3
"""
psd_dashboard_full.py

Standalone script:
 - Reads three NetCDF files (analysis, forecast, prediction)
 - Detects common 2D variables (time, lat, lon)
 - Precomputes PSDs for each (variable, season, metric, dim, model) at multiple resolutions
 - Saves a self-contained HTML dashboard with side-by-side model comparison and interactive selectors.

Edit INPUT_* paths below to point to your files.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import welch
import holoviews as hv
import panel as pn

hv.extension("bokeh")
pn.extension()

from earthml.utils import Dask, XarrayDataset
from datetime import datetime, timedelta
from pathlib import Path, PosixPath
import joblib, torch, os

# -------------------------
# USER CONFIG — EDIT PATHS
# -------------------------
REGION = "ConUS"
TRAIN_PERIOD = "20210323-20231231"
TEST_PERIOD = "20240101-20240531"
EXP_SUFFIX = "_32bs_juno"
# EXP_SUFFIX = "_32bs_era5"

OUT_HTML = "psd_dashboard.html"

VARIABLES = {'msl', 'u10', 'd2m', 't2m', 'v10', 'tcc'}

# spectral parameters (progressive multi-resolution set here)
RESOLUTIONS = [64, 128, 256, 512]   # nperseg values to compute
FS = 1/0.1                          # sampling frequency (reciprocal of resolution)
NOVERLAP = None                     # uses default (0) if None; or set e.g. int(nperseg/2)

# seasons and metrics to compute
SEASONS = ["All", "DJF", "MAM", "JJA", "SON"]
METRICS = ["Mean", "Variance", "RMSE"]   # RMSE computed vs. temporal mean of the period (as simple ref)

# spatial dims naming in your dataset
LAT_NAME_CANDIDATES = ["lat", "latitude", "y"]
LON_NAME_CANDIDATES = ["lon", "longitude", "x"]
TIME_NAME_CANDIDATES = ["valid_time", "times", "time"]

# -------------------------
# Helper utilities
# -------------------------

def find_coord(ds, candidates):
    for c in candidates:
        if c in ds.dims or c in ds.coords:
            return c
    raise KeyError(f"None of {candidates} found in dataset dims/coords.")

def season_mask_from_time(time_index, season):
    months = pd.to_datetime(time_index).month
    if season == "All":
        return np.ones_like(months, dtype=bool)
    if season == "DJF":
        return np.isin(months, [12,1,2])
    if season == "MAM":
        return np.isin(months, [3,4,5])
    if season == "JJA":
        return np.isin(months, [6,7,8])
    if season == "SON":
        return np.isin(months, [9,10,11])
    return np.ones_like(months, dtype=bool)

def safe_wavelengths(freqs):
    # convert freqs -> wavelengths, remove zeros and negative / non-finite
    with np.errstate(divide='ignore', invalid='ignore'):
        wl = np.where(freqs > 0, 1.0 / freqs, np.nan)
    # Remove NaN or inf
    mask = np.isfinite(wl)
    wl = wl[mask]
    return wl, mask

def ensure_increasing(x, arrays_to_flip=None):
    """Ensure x is strictly increasing; if decreasing, reverse x and corresponding arrays."""
    arrays_to_flip = arrays_to_flip or []
    if len(x) == 0:
        return x, arrays_to_flip
    if x[0] > x[-1]:
        x = x[::-1]
        arrays_to_flip = [a[::-1] for a in arrays_to_flip]
    return x, arrays_to_flip

def powerspectrum_welch_2d (self, data, fsx, fsy):
        """
        Compute power spectra using Welch's method.
        Parameters:
        - data: 2D array of values (lat, lon) or (time, lat) or (time, lon)
        - fs: sampling frequency in degrees
        Returns:
        - fxx, fyy: wavenumbers
        - Pxy, Pyx: power spectral densities along y, x for all x, y
        """
        from scipy.signal import welch
        # Welch spectral analysis params, noverlap < nperseg
        nperseg, noverlap, nfft = 50, 30, 100
        Pxy, Pyx = [], []
        # Compute Welch's power spectra
        for i in range(data.shape[0]): # ps along x axis at fixed y for all y's
            fyy, Py = welch(data[i, :], fsy, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            Pyx.append(Py[1:])
        for j in range(data.shape[1]): # ps along y axis at fixed x for all x's
            fxx, Px = welch(data[:, j], fsx, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            Pxy.append(Px[1:])
        self.logger.debug(f"Welch x idx {j}: fxx {fxx}")
        self.logger.debug(f"Welch y idx {i}: fyy {fyy}")
        return fxx[1:], fyy[1:], np.array(Pxy), np.array(Pyx)

if __name__ == "__main__":
    dask_earthml = Dask()
    client, cluster = dask_earthml.client, dask_earthml.cluster
    print("Dask dashboard:", client.dashboard_link)
    # -------------------------
    # Load datasets & detect variables
    # -------------------------
    print("Loading datasets...")

    exp_root_folder = "/work/cmcc/jd19424/test-ML/experiments_earthML/"
    ds_pr, ds_fc, ds_an = None, None, None
    for v in VARIABLES:
        exp_name = f"exp_{v}-{REGION}-{TRAIN_PERIOD}_{v}-{REGION}-{TEST_PERIOD}"
        exp_path = Path(exp_root_folder).joinpath(exp_name+EXP_SUFFIX+"/experiment.cfg")
        print(f"Experiment path: {exp_path}")
        experiment = joblib.load(exp_path)
        # print(experiment)
        pr_v = experiment['test_data']['prediction'].load()
        fc_v = experiment['test_data']['input'].load()
        an_v = XarrayDataset.regrid(fc_v, experiment['test_data']['target'].load()) # regrid only if necessary
        ds_pr = xr.merge([ds_pr, pr_v], compat='no_conflicts') if ds_pr else pr_v
        ds_fc = xr.merge([ds_fc, fc_v], compat='no_conflicts') if ds_fc else fc_v
        ds_an = xr.merge([ds_an, an_v], compat='no_conflicts') if ds_an else an_v

    # find standard coord names
    time_name = find_coord(ds_an, TIME_NAME_CANDIDATES)
    lat_name = find_coord(ds_an, LAT_NAME_CANDIDATES)
    lon_name = find_coord(ds_an, LON_NAME_CANDIDATES)

    # # variable detection: numeric variables present in all three datasets and containing time,lat,lon
    # def detect_common_vars(ds_list):
    #     sets = [set([v for v in ds.data_vars if set(ds[v].dims) >= {time_name, lat_name, lon_name}]) for ds in ds_list]
    #     common = set.intersection(*sets)
    #     return sorted(list(common))

    # variables = detect_common_vars([ds_an, ds_fc, ds_pr])
    # if not variables:
    #     raise RuntimeError("No common time-lat-lon variables found across the three datasets.")

    print("Variables:", set(ds_fc.data_vars))

    # spatial coords
    lats = ds_an[lat_name].values
    lons = ds_an[lon_name].values
    times = ds_an[time_name].values

    # -------------------------
    # Precompute PSDs
    # -------------------------
    print("Starting precomputation...")
    precomputed = {}  # key: (var, season, metric, dim, nperseg, model) -> dict with freqs, spatial_coords, PSD

    models = {
        "Analysis": ds_an,
        "Forecast": ds_fc,
        "Prediction": ds_pr
    }

    # iterate
    for var in VARIABLES:
        print(f"Variable: {var}")
        # load arrays into memory to be fast (careful with huge datasets)
        arrs = {m: models[m][var].values for m in models}
        for season in SEASONS:
            mask_time = season_mask_from_time(times, season)
            for metric in METRICS:
                # compute "field" per model for mean/var/RMSE along time prior to spatial-welch reduction
                # For Mean/Variance we compute the temporal mean/var first -> yields 2D field (lat,lon)
                # For RMSE we'll base on temporal RMSE vs period mean (simple choice)
                # We'll compute PSDs along the chosen spatial dimension by sliding through the other spatial axis
                    for nperseg in RESOLUTIONS:
                        nover = int(nperseg/2) if NOVERLAP is None else NOVERLAP
                        for model_name, arr in arrs.items():
                            fxx, fyy, Pxy, Pyx = powerspectrum_welch_2d(arr, FS, FS)
                            Pxy, Pyx = np.log(Pxy), np.log(Pyx)
                            Pxy_min, Pyx_min = (np.nanmin(Pxy), np.nanmin(Pyx))
                            Pxy_max, Pyx_max = (np.nanmax(Pxy), np.nanmax(Pyx))
                            Pxy_center = 0 if Pxy_min < 0 and Pxy_max > 0 else Pxy_min+(Pxy_max-Pxy_min)/2
                            Pyx_center = 0 if Pyx_min < 0 and Pyx_max > 0 else Pyx_min+(Pyx_max-Pyx_min)/2
                            # # arr shape: (time, lat, lon)
                            # arr_s = arr[mask_time]  # time-sliced
                            # # For Mean / Var we reduce in time first and then compute PSD along spatial dim across the other axis
                            # if metric == "Mean":
                            #     field = np.mean(arr_s, axis=0)   # shape (lat, lon)
                            # elif metric == "Variance":
                            #     field = np.var(arr_s, axis=0)
                            # elif metric == "RMSE":
                            #     # simple RMSE vs temporal mean (could be adjusted to different reference)
                            #     clim = np.mean(arr_s, axis=0)
                            #     field = np.sqrt(np.mean((arr_s - clim[np.newaxis, ...])**2, axis=0))
                            # else:
                            #     raise ValueError("Unknown metric")

                            # # Build PSD array: for each index along the OTHER axis compute PSD along chosen dim
                            # # e.g., if dim=='latitude' (axis=0), then for each longitude index j compute PSD over latitude array field[:, j]
                            # n_other = field.shape[1 - axis]
                            # psd_rows = []
                            # freqs = None
                            # for j in range(n_other):
                            #     if axis == 0:
                            #         sig = field[:, j]
                            #     else:
                            #         sig = field[j, :]

                            #     # if signal is constant or contains NaNs, welch may give zeros or NaNs; handle gracefully
                            #     # replace infs with nan
                            #     sig = np.asarray(sig, dtype=float)
                            #     if not np.isfinite(sig).all():
                            #         sig = np.nan_to_num(sig, nan=np.nanmean(sig[np.isfinite(sig)]) if np.isfinite(sig).any() else 0.0)

                            #     # handle signals too short for nperseg
                            #     if sig.size < 2:
                            #         f = np.array([np.nan])
                            #         Pxx = np.array([np.nan])
                            #     else:
                            #         # Use scipy.signal.welch
                            #         try:
                            #             f, Pxx = welch(sig, fs=FS, nperseg=min(nperseg, sig.size), noverlap=min(nover, sig.size-1) if sig.size>1 else 0)
                            #         except Exception:
                            #             # fallback - produce NaNs
                            #             f = np.array([np.nan])
                            #             Pxx = np.array([np.nan])

                            #     freqs = f
                            #     psd_rows.append(Pxx)

                            # psd_rows = np.array(psd_rows)  # shape (n_other, n_freq)
                for dim in ["latitude", "longitude"]:
                    # set axis mapping for slicing when computing 1D signals across chosen dim
                    axis = 0 if dim == "latitude" else 1
                    spatial_coords = lats if dim == "latitude" else lons
                            # Store
                            key = (var, season, metric, dim, nperseg, model_name)
                            precomputed[key] = {
                                "freqs": freqs,
                                "spatial": spatial_coords,
                                "psd": psd_rows  # shape (n_space, n_freq) where n_space == n_other
                            }

    print("Precomputation finished. Keys stored:", len(precomputed))

    # -------------------------
    # Dashboard building (no recompute)
    # -------------------------
    print("Building dashboard UI...")

    # Widgets
    var_widget = pn.widgets.Select(name="Variable", options=list(VARIABLES), value=list(VARIABLES)[0])
    season_widget = pn.widgets.Select(name="Season", options=SEASONS, value="All")
    metric_widget = pn.widgets.Select(name="Metric", options=METRICS, value=METRICS[0])
    dim_widget = pn.widgets.Select(name="Spectral Dimension", options=["latitude", "longitude"], value="latitude")

    # resolution selector: allow "All" to overlay multiple PSDs of different resolutions
    res_options = ["All"] + [str(r) for r in RESOLUTIONS]
    res_widget = pn.widgets.Select(name="Resolution (nperseg)", options=res_options, value=str(RESOLUTIONS[1]))

    relative_toggle = pn.widgets.Checkbox(name="Relative power (normalize each spectrum)", value=True)
    display_wavelength = pn.widgets.Checkbox(name="Display as wavelength (instead of frequency)", value=False)
    overlay_opacity = pn.widgets.FloatSlider(name="Overlay opacity (when 'All')", start=0.1, end=1.0, step=0.05, value=0.6)

    def make_image_from_psd(freqs, spatial, psd_2d, show_wavelength=False, normalize=False):
        """
        Convert freqs (n_freq,), spatial (n_space,), psd_2d (n_space, n_freq) into a Holoviews Image.
        Returns (hv.Image or hv.Text, x_axis_label)
        """
        freqs = np.asarray(freqs)
        spatial = np.asarray(spatial)
        arr = np.array(psd_2d)   # shape (n_space, n_freq)

        if freqs is None or freqs.size == 0:
            return hv.Text(0, 0, "No data"), "frequency"

        # Choose axis (frequency or wavelength) and select valid columns
        if show_wavelength:
            wl, mask = safe_wavelengths(freqs)
            if wl.size == 0:
                x = freqs
                x_label = "frequency"
                arr_sel = arr
            else:
                x = wl
                arr_sel = arr[:, mask]   # keep shape (n_space, n_wl)
                x_label = "wavelength"
        else:
            mask = np.isfinite(freqs)
            x = freqs[mask]
            arr_sel = arr[:, mask]
            x_label = "frequency"

        # Normalize each spatial row if requested (row-wise normalization)
        if normalize:
            row_sums = np.nansum(arr_sel, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            arr_sel = arr_sel / row_sums

        # Ensure x increasing AND flip corresponding columns of arr_sel if needed
        x, arr_cols = ensure_increasing(x, arrays_to_flip=[arr_sel])
        arr_sel = arr_cols[0]

        # Ensure spatial increasing AND flip corresponding rows of arr_sel if needed
        spatial, arr_rows = ensure_increasing(spatial, arrays_to_flip=[arr_sel])
        arr_sel = arr_rows[0]

        # Now arr_sel shape is (n_space, n_x) where n_space == len(spatial), n_x == len(x)
        print("x.shape", x.shape, "spatial.shape", spatial.shape, "arr_sel.shape", arr_sel.shape)
        try:
            img = hv.Image((x, spatial, arr_sel), kdims=[x_label, "space"], vdims=["power"])
            return img, x_label
        except Exception as e:
            # raise a proper exception so the outer code can catch it / show meaningful message
            raise RuntimeError(f"Plot error creating hv.Image: {e}")


    def build_side_by_side(var, season, metric, dim, resolution_selection, normalize, show_wl, overlay_opacity_val):
        """
        Build 3-panel comparison: Analysis | Forecast | Prediction
        resolution_selection can be 'All' or string of integer from RESOLUTIONS
        """
        panels = []
        titles = ["Analysis", "Forecast", "Prediction"]
        for model_name in titles:
            if resolution_selection == "All":
                # overlay all resolutions (semi-transparent)
                imgs = []
                for nperseg in RESOLUTIONS:
                    key = (var, season, metric, dim, nperseg, model_name)
                    if key not in precomputed:
                        continue
                    d = precomputed[key]
                    freqs = d["freqs"]
                    spatial = d["spatial"]
                    psd = d["psd"]  # shape (n_space, n_freq)
                    print("DEBUG:", var, season, metric, dim, resolution_selection)
                    print("freqs:", freqs.shape)
                    print("spatial (coords):", spatial.shape)
                    print("psd (raw):", psd.shape)
                    img, xlab = make_image_from_psd(freqs, spatial, psd, show_wavelength=show_wl, normalize=normalize)
                    if isinstance(img, hv.Image):
                        imgs.append(img.opts(alpha=overlay_opacity_val, line_width=0))
                if not imgs:
                    panels.append(hv.Text(0,0,"No data"))
                else:
                    # overlay all images (they share axes)
                    combo = None
                    for im in imgs:
                        combo = im if combo is None else (combo * im)
                    # add title
                    combo = combo.relabel(model_name)
                    panels.append(combo)
            else:
                nperseg = int(resolution_selection)
                key = (var, season, metric, dim, nperseg, model_name)
                if key not in precomputed:
                    panels.append(hv.Text(0,0,"No data"))
                else:
                    d = precomputed[key]
                    freqs = d["freqs"]
                    spatial = d["spatial"]
                    psd = d["psd"]
                    print("DEBUG:", var, season, metric, dim, resolution_selection)
                    print("freqs:", freqs.shape)
                    print("spatial (coords):", spatial.shape)
                    print("psd (raw):", psd.shape)
                    img, xlab = make_image_from_psd(freqs, spatial, psd, show_wavelength=show_wl, normalize=normalize)
                    if isinstance(img, hv.Image):
                        img = img.opts(
                            cmap="viridis",
                            colorbar=True,
                            width=350,
                            height=350,
                            xlabel=("Wavelength" if show_wl else "Frequency"),
                            ylabel=(dim),
                        )
                    panels.append(img.relabel(model_name) if isinstance(img, hv.Image) else img)

        # arrange side-by-side
        return hv.Layout(panels).cols(3)

    @pn.depends(var_widget, season_widget, metric_widget, dim_widget, res_widget, relative_toggle, display_wavelength, overlay_opacity)
    def view(var, season, metric, dim, resolution_selection, normalize, show_wl, overlay_opacity_val):
        # return the HoloViews layout embedded in a Panel pane
        hv_plot = build_side_by_side(var, season, metric, dim, resolution_selection, normalize, show_wl, overlay_opacity_val)
        # if wavelength mode, use xscale log (makes sense) -- use opts on hv_plot
        xscale = 'log' if show_wl else 'linear'
        xscale_bool = (xscale == 'log')
        return pn.Row(
            hv_plot.options({
                "Image": {"logx": xscale_bool},
                "QuadMesh": {"logx": xscale_bool},
                "Curve": {"logx": xscale_bool},
            }),
            sizing_mode="stretch_width"
        )


    # assemble dashboard layout
    controls = pn.Row(var_widget, season_widget, metric_widget, dim_widget, res_widget)
    toggles = pn.Row(relative_toggle, display_wavelength, overlay_opacity)

    dashboard = pn.Column(
        pn.pane.Markdown("# PSD Dashboard — Precomputed (Analysis | Forecast | Prediction)"),
        pn.pane.Markdown("**Instructions:** Select variable / season / metric / spectral dimension. "
                        "Choose a resolution or *All* to overlay multiple resolutions. Toggle relative power or wavelength display."),
        controls,
        toggles,
        pn.panel(view, sizing_mode="stretch_both")
    )

    # -------------------------
    # Save standalone HTML
    # -------------------------
    print(f"Saving dashboard to {OUT_HTML} (this may take a moment)...")
    dashboard.save(OUT_HTML, embed=True)
    print("Saved. Open the generated HTML in your browser.")
