import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec
import matplotlib.pyplot as plt
import hvplot.xarray
import cartopy.crs as ccrs
from scipy.ndimage import gaussian_filter
from earthml.utils import Dask, XarrayDataset
from pathlib import Path
import joblib, os

# USER CONFIG
REGION = "ConUS"
TRAIN_PERIOD = "20210323-20231231"
TEST_PERIOD = "20240101-20240531"
EXP_SUFFIX = "_32bs_juno"
VAR = 't2m'
OUTPUT_DIR = "/work/cmcc/jd19424/test-ML/dataML/filtered_data"
FILENAME = "smoothed_t2m"

if __name__ == "__main__":
    # Load data
    dask_earthml = Dask()
    client, cluster = dask_earthml.client, dask_earthml.cluster
    print("Dask dashboard:", client.dashboard_link)
    
    print("Loading datasets...")
    exp_root_folder = "/work/cmcc/jd19424/test-ML/experiments_earthML/"
    exp_name = f"exp_{VAR}-{REGION}-{TRAIN_PERIOD}_{VAR}-{REGION}-{TEST_PERIOD}"
    exp_path = Path(exp_root_folder).joinpath(exp_name+EXP_SUFFIX+"/experiment.cfg")
    
    print(f"Experiment path: {exp_path}")
    experiment = joblib.load(exp_path)
    
    ds_fc_train = experiment['train_data']['input'].load()
    ds_an_train = experiment['train_data']['target'].load()
    ds_fc_test = experiment['test_data']['input'].load()
    ds_an_test = experiment['test_data']['target'].load()
    
    for spatial_da, var_type in zip(
        [ds_fc_train[VAR], ds_an_train[VAR], ds_fc_test[VAR], ds_an_test[VAR]],
        ['train_input', 'train_target', 'test_input', 'test_target']
    ):
        print(f"Generate filtered {VAR} data for {var_type}")
        # Set Parameters for Progressive Smoothing
        # 1. Physical Constants (Approximation near the equator/mid-latitudes)
        DEGREE_TO_KM = 111.0  # 1 degree is approx 111 km
        DATA_RESOLUTION_DEG = 0.1
        DATA_RESOLUTION_KM = DATA_RESOLUTION_DEG * DEGREE_TO_KM # ~11.1 km

        # 2. Define smoothing levels based on desired physical radius (R)
        R_steps_km = [25.0, 50.0, 100.0, 200.0]  # Radius in kilometers

        # 3. Convert R (km) to sigma (grid units) using R ≈ 3*sigma
        # sigma_grid = R_km / (3 * Resolution_km)
        sigma_steps = np.array(R_steps_km) / (3.0 * DATA_RESOLUTION_KM)

        # Convert to Python list for script usage
        sigma_steps = sigma_steps.tolist() 

        print(f"Desired Radii (km): {R_steps_km}")
        print(f"Calculated Sigma (grid units): {sigma_steps}") 

        # Set Parameters for Progressive Smoothing

        # The calculated sigma values are now based on a physical radius.
        # Example: sigma=1.5 will smooth features smaller than ~50 km
        print(f"Applying progressive 2D Gaussian smoothing with sigma values: {sigma_steps}")

        # Filtering and Xarray Creation
        smoothed_data_list = []

        # Dynamic Sigma Logic:
        # We assume the LAST two dimensions are spatial (e.g., Lat, Lon) and should be smoothed.
        # Any preceding dimensions (Time, Member, Batch) get sigma=0 (no smoothing).
        # Example 3D: (Time, Lat, Lon) -> sigma=(0, s, s)
        # Example 2D: (Lat, Lon)       -> sigma=(s, s)
        ndims = spatial_da.ndim
        
        if ndims < 2:
            raise ValueError(f"Data array must have at least 2 spatial dimensions, found {ndims}")

        for R, sigma in zip(R_steps_km, sigma_steps):
            # 1. Construct sigma tuple dynamically
            # [0] * (3-2) -> [0] ... + [s, s] -> [0, s, s]
            sigma_params = [0] * (ndims - 2) + [sigma, sigma]
            
            # 2. Apply filter
            data_filtered = gaussian_filter(spatial_da.values, sigma=sigma_params, mode='reflect')

            # 3. Create Xarray DataArray
            var_name = f'{VAR}_smoothed_R{R}_s{sigma:.1f}'
            da_smoothed = xr.DataArray(
                data_filtered,
                coords=spatial_da.coords,
                dims=spatial_da.dims,
                name=var_name,
                attrs={'long_name': f'Smoothed radius {R} (sigma={sigma:.1f})', 'sigma_value': sigma, 'radius': R}
            )
            smoothed_data_list.append(da_smoothed)

        # Final Xarray Dataset Creation
        ds_spatial = xr.Dataset(
            data_vars={da.name: da for da in smoothed_data_list},
            coords=spatial_da.coords
        )
        # Add original
        ds_spatial[f'{VAR}_original'] = spatial_da

        print("--- Summary of Generated Dataset ---")
        print(ds_spatial)

        # Save dataset in Zarr
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        encoding_zarr = (
            {v: {"compressors": compressor} for v in list(ds_spatial.data_vars)}
        )
        store = Path(OUTPUT_DIR).joinpath(Path(f"{FILENAME}_{var_type}").with_suffix(".zarr"))
        print(f"Save to {store}")
        ds_spatial.to_zarr(store, encoding=encoding_zarr, mode='w', consolidated=False)

        # Plots
        print("--- Generating Time-Averaged Plots ---")
        
        # Create output directory
        out_path = Path(OUTPUT_DIR)
        out_path.mkdir(exist_ok=True, parents=True)

        # 1. Compute Time Mean
        if 'valid_time' in ds_spatial.dims:
            print("Computing time mean...")
            ds_mean = ds_spatial.mean(dim='valid_time', keep_attrs=True)
        else:
            ds_mean = ds_spatial

        # 2. Save Static Plots (Matplotlib)
        for var_name in ds_mean.data_vars:
            # a. Define the Coordinate Reference System (CRS) for the plot
            # Use Plate Carree (ccrs.PlateCarree) for a standard lat/lon map
            fig, ax = plt.subplots(
                figsize=(10, 6), 
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            
            # b. Plot the data
            ds_mean[var_name].plot.pcolormesh( 
                ax=ax, 
                x='longitude', 
                y='latitude', 
                transform=ccrs.PlateCarree(),
                cmap='viridis', 
                robust=True,
                add_colorbar=True
            )
            
            # c. Add geographical features for context
            ax.coastlines(resolution='50m', color='black', linewidth=0.8)
            ax.set_title(f"Time-Averaged {var_name} {var_type}")
            
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            
            plt.tight_layout()
            
            filename = out_path / f"{var_name}_{var_type}_avg.png"
            plt.savefig(filename, dpi=150)
            print(f"Saved: {filename}")
            plt.close(fig)

        # 3. Save Interactive Comparison with Selector (Hvplot)
        try:
            print("Generating interactive HTML comparison with selector...")
            # a. Create a list of DataArrays
            da_list = [ds_mean[v] for v in ds_mean.data_vars]
            
            # b. Assign the variable names as the values for the new selector dimension
            smoothing_levels = list(ds_mean.data_vars)
            
            # c. Concatenate into a new DataArray with a new dimension named 'smoothing_level'
            ds_combined = xr.concat(da_list, dim='smoothing_level')
            ds_combined['smoothing_level'] = smoothing_levels
            
            # d. Use hvplot to plot the new combined DataArray, faceting by the new dimension
            plot = ds_combined.hvplot.quadmesh(
                x='longitude', y='latitude',
                cmap='viridis',
                projection=ccrs.PlateCarree(),
                geo=True,
                coastline=True,
                project=True,
                # Use the new dimension for the selector widget
                groupby='smoothing_level', 
                title=f"Time-Averaged Smoothing Comparison ({VAR} {var_type})",
                clabel="Temperature (K)" # Optional: improve colorbar label
            )
            
            html_filename = out_path / f"smoothing_comparison_{VAR}_{var_type}.html"
            hvplot.save(plot, html_filename)
            print(f"Saved interactive plot with selector: {html_filename}")
            
        except Exception as e:
            print(f"Skipping hvplot generation: {e}")
            print(f"Details: {e}")

    print("Done.")
    