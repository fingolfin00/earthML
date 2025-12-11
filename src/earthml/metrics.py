from typing import List, Set

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

class Metrics:
    def __init__ (self, truth: xr.Dataset, data: xr.Dataset | List[xr.Dataset], truth_name: str , data_name: str | List[str]):
        self.truth = truth
        self.data = data if isinstance(data, list) else [data]
        self.truth_name = truth_name
        self.data_name = data_name if isinstance(data_name, list) else [data_name]
        self.time_dim, self.spatial_dims = self._get_dim()

    def _get_dim (self):
        # TODO add check for variable consistency along all datasets (e.g. same number of variables)
        truth_vars = set(self.truth)
        time_dim = self.truth.cf['time'].name
        spatial_dims = [dim for dim in self.truth.dims if dim != time_dim]
        for i, d in enumerate(self.data):
            d_vars = list(d.data_vars)
            if set(d_vars) != set(truth_vars):
                print(f"Renaming {self.data_name[i]} vars to match {self.truth_name}")
                rename_map = {old: new for old, new in zip(d_vars, truth_vars)}
                self.data[i] = d.rename_vars(rename_map)
        for i, d in enumerate(self.data):
            time_dim_current = d.cf['time'].name
            if time_dim_current != time_dim:
                self.data[i] = d.rename_dims({time_dim_current: time_dim})
        print(f"Time dimension: {time_dim}, spatial dimensions: {spatial_dims}, vars: {truth_vars}") # TODO add real check on vars
        return time_dim, spatial_dims

    # Metrics (a: truth, b: data)
    @staticmethod
    def rmse(a, b, time_dim): return ((a - b)**2).mean(dim=time_dim).pipe(np.sqrt)
    @staticmethod
    def mae(a, b, time_dim): return abs(a - b).mean(dim=time_dim)
    @staticmethod
    def bias(a, b, time_dim): return (a - b).mean(dim=time_dim)
    @staticmethod
    def mape(a, b, time_dim, eps): return (abs((a - b) / (a + eps)) * 100).mean(dim=time_dim)
    @staticmethod
    def stderr(a, b, time_dim): return (a - b).std(dim=time_dim)
    @staticmethod
    def r2(a, b, time_dim, eps):
        sst = ((a - a.mean(dim=time_dim))**2).sum(dim=time_dim)
        sse = ((a - b)**2).sum(dim=time_dim)
        return 1 - sse / (sst + eps)
    @staticmethod
    def fss(a, b, time_dim, threshold, eps):
        """Fraction Skill Score"""
        A, B = (a > threshold).astype(float), (b > threshold).astype(float)
        return 1 - ((A - B)**2).mean(dim=time_dim) / ((A**2 + B**2).mean(dim=time_dim) + eps)
    @staticmethod
    def sal(a, b, time_dim, eps): #TODO better understand this metric
        """Std ratio, Avg (mean) ratio, weighted mean difference (L)"""
        return {
            "S": (b.std(dim=time_dim) - a.std(dim=time_dim)) / (a.std(dim=time_dim) + eps),
            "A": (b.mean(dim=time_dim) - a.mean(dim=time_dim)) / (a.mean(dim=time_dim) + eps),
            "L": (b.weighted(abs(b)).mean(dim=time_dim) - a.weighted(abs(a)).mean(dim=time_dim)) # TODO probably wrong, check weights
        }
    @staticmethod
    def spectral_metrics(a, b, time_dim, spatial_dims, eps):
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

    def compute_spatial_metrics(self, fss_threshold=0.5, eps=1e-6):
        def _geo_avg (data): #TODO maybe promote to upper level class method
            """Compute geographically weighted mean"""
            # Weights for latitude-averaged mean
            lat = data.cf['latitude']
            geo_weights = np.cos(np.deg2rad(lat))
            return data.weighted(geo_weights).mean(dim=self.spatial_dims).compute().values

        def _aggregate(da):
            """Spatial aggregation wrapper"""
            ts = _geo_avg(da)
            return {
                "global_mean": ts.mean(),
                "time_series": ts,
                "mean_lat": da.mean(self.spatial_dims[1]).values if len(self.spatial_dims) == 2 else None,
                "mean_lon": da.mean(self.spatial_dims[0]).values if len(self.spatial_dims) == 2 else None,
            }
        def compute_set(a, b):
            """Compute metrics per variable"""
            out = {}
            for v in a.data_vars:
                av, bv = a[v], b[v]
                out[v] = {
                    "rmse": self.rmse(av, bv, self.time_dim),
                    "mae": self.mae(av, bv, self.time_dim),
                    "bias": self.bias(av, bv, self.time_dim),
                    "mape": self.mape(av, bv, self.time_dim, eps),
                    "stderr": self.stderr(av, bv, self.time_dim),
                    "r2": self.r2(av, bv, self.time_dim, eps),
                    "fss": self.fss(av, bv, self.time_dim, fss_threshold, eps),
                    # dictionaries
                    "sal": self.sal(av, bv, self.time_dim, eps),
                    "spectral": self.spectral_metrics(av, bv, self.time_dim, self.spatial_dims, eps),
                }
                # add aggregated means
                out[v]["agg"] = {
                    k: _aggregate(out[v][k]) 
                    for k in ["rmse", "mae", "bias", "mape", "stderr", "r2", "fss"]
                }

            return out

        return {name: compute_set(self.truth, d) for d, name in zip(self.data, self.data_name)}

    def print_aggregated_metrics(self, metrics):
        """
        Print a tabular comparison of average metrics between FC and PR models.
        
        Parameters:
        -----------
        metrics : dict
            Output from compute_spatial_metrics function
        """
        # Support only TWO data comparable with truth for now
        assert len(self.data) <= 2
        data_name1 = self.data_name[0] # usually forecast
        data_name2 = self.data_name[1] if len(self.data_name) > 1 else None
        # Extract all variables
        chosen_data_name = list(metrics.keys())[0]
        chosen_var = list(metrics[chosen_data_name].keys())[0]
        variables = list(metrics[chosen_data_name].keys())
        
        # Metric names to compare (only aggregated)
        metric_names = list(metrics[chosen_data_name][chosen_var]['agg'].keys())
        print(f"Aggregated metrics: {metric_names}")
        
        # Build comparison table for each variable
        for var in variables:
            print(f"\n{'='*70}")
            print(f"Variable: {var}")
            print(f"{'='*70}")
            
            # Create rows for the table
            rows = []
            for metric in metric_names:
                val1 = metrics[data_name1][var]['agg'][metric]['global_mean'] # forecast
                val2 = metrics[data_name2][var]['agg'][metric]['global_mean'] if data_name2 else None

                diff = val2 - val1 if data_name2 else None
                abs_diff = abs(val2) - abs(val1) if data_name2 else None
                pct_abs_diff = ((abs(val2) - abs(val1)) / (abs(val1) + 1e-10)) * 100 if data_name2 else None
                
                rows.append({
                    'Metric': metric.replace(' avg', '').upper(),
                    data_name1 : f"{val1:.4f}",
                    data_name2: f"{val2:.4f}" if data_name2 else "",
                    f'Diff ({data_name2}-{data_name1})': f"{diff:+.4f}" if data_name2 else "",
                    f'Abs diff': f"{abs_diff:+.4f}" if data_name2 else "",
                    'Abs diff [%]': f"{pct_abs_diff:+.2f}%" if data_name2 else "",
                })
            
            # Create and print DataFrame
            df = pd.DataFrame(rows)
            print(df.to_string(index=False))

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
