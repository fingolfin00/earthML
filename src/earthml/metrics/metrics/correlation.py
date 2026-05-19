from typing import Literal, Sequence
import numpy as np
import xarray as xr

from .base import BaseMetrics


ClimType = Literal["month", "dayofyear", "season"]


class CorrelationMetrics(BaseMetrics):
    def __init__(
        self,
        truth_data: xr.Dataset,
        model_data: xr.Dataset | list[xr.Dataset],
        truth_name: str,
        model_names: str | list[str],
        clim_data: xr.Dataset | list[xr.Dataset] | None = None,
        mask_data: xr.Dataset | None = None,
    ):
        super().__init__(
            truth_data=truth_data,
            model_data=model_data,
            truth_name=truth_name,
            model_names=model_names,
            clim_data=clim_data,
            mask_data=mask_data, # not used
        )


    def _corr_da(
        self,
        x: xr.DataArray,
        y: xr.DataArray,
        corr_reduce_dims: str | Sequence[str],
        min_periods: int = 2,
        eps: float = 1e-8,
    ) -> xr.DataArray:
        """
        Compute weighted Pearson correlation between two DataArrays.

        Applies pairwise-valid masking, enforces a minimum number of samples,
        and uses geo-weighted covariance and standard deviation.

        Parameters
        ----------
        x, y : xr.DataArray
            Input arrays to correlate.
        corr_reduce_dims : str or sequence of str
            Dimension(s) over which to compute the correlation.
        min_periods : int, optional
            Minimum number of valid paired samples required.
        eps : float, optional
            Threshold to avoid division by near-zero standard deviation.

        Returns
        -------
        xr.DataArray
            Correlation values with reduced dimensions removed.
        """
        pairwise_valid = np.isfinite(x) & np.isfinite(y)
        x = x.where(pairwise_valid)
        y = y.where(pairwise_valid)

        n = pairwise_valid.sum(dim=corr_reduce_dims)

        # weighted covariance consistent with geo_mean
        cov = x.earthml.geo_cov(y, corr_reduce_dims)
        std_x = x.earthml.geo_std(corr_reduce_dims)
        std_y = y.earthml.geo_std(corr_reduce_dims)

        valid = (
            (n >= min_periods)
            & np.isfinite(cov)
            & np.isfinite(std_x)
            & np.isfinite(std_y)
            & (std_x > eps)
            & (std_y > eps)
        )
        denom = (std_x * std_y).where(valid)
        return cov.where(valid) / denom


    def corr(
        self,
        corr_reduce_dims: str | Sequence[str],
        min_periods: int = 2,
        eps: float = 1e-8,
    ) -> xr.Dataset:
        """
        Compute Pearson correlation between each model and truth dataset.

        Correlation is computed independently for each variable and model,
        reducing over the specified dimensions.

        Parameters
        ----------
        corr_reduce_dims : str or sequence of str
            Dimension(s) over which to compute correlation.
        min_periods : int, optional
            Minimum number of valid paired samples required.
        eps : float, optional
            Threshold to avoid division by near-zero standard deviation.

        Returns
        -------
        xr.Dataset
            Dataset of correlation values with a 'model' dimension.
        """
        corr_reduce_dims = self._as_tuple(corr_reduce_dims)
        
        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            return self._corr_da(
                model,
                truth,
                corr_reduce_dims=corr_reduce_dims,
                min_periods=min_periods,
                eps=eps,
            )

        return self._apply_metric_per_model(metric_func)


    def corr_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        corr_reduce_dims: str | Sequence[str],
        min_periods: int = 2,
        eps: float = 1e-8,
    ) -> xr.Dataset:
        """
        Compute Pearson correlation between mean fields of model and truth.

        The data are first averaged over `data_mean_dims`, then correlation is
        computed over `corr_reduce_dims`.

        Parameters
        ----------
        data_mean_dims : str or sequence of str
            Dimension(s) over which to compute the mean fields.
        corr_reduce_dims : str or sequence of str
            Dimension(s) over which to compute correlation.
        min_periods : int, optional
            Minimum number of valid paired samples required.
        eps : float, optional
            Threshold to avoid division by near-zero standard deviation.

        Returns
        -------
        xr.Dataset
            Dataset of correlation values with a 'model' dimension.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        corr_reduce_dims = self._as_tuple(corr_reduce_dims)

        self._check_dims_overlap(
            data_mean_dims, corr_reduce_dims, "data_mean_dims", "corr_reduce_dims"
        )

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)
            return self._corr_da(
                model_mean,
                truth_mean,
                corr_reduce_dims=corr_reduce_dims,
                min_periods=min_periods,
                eps=eps,
            )

        return self._apply_metric_per_model(metric_func)


    def _climatology_anom(
        self,
        ds: xr.Dataset | xr.DataArray,
        clim_ds: xr.Dataset | xr.DataArray | None,
        period: ClimType = "month",
        min_periods: int = 2,
    ) -> xr.Dataset | xr.DataArray:
        """
        Compute anomalies relative to a climatology.

        Climatology can be computed from the input data or provided externally.
        Grouping is performed along the time dimension using the specified period.

        Parameters
        ----------
        ds : xr.Dataset or xr.DataArray
            Input data.
        clim_ds : xr.Dataset or xr.DataArray or None
            External climatology. If None, computed from `ds`.
        period : {"month", "dayofyear", "season"}, optional
            Temporal grouping used to compute climatology.
        min_periods : int, optional
            Minimum number of samples required per group.

        Returns
        -------
        xr.Dataset or xr.DataArray
            Anomaly field with same structure as input.
        """
        if period == "month":
            group = f"{self.dims.time}.month"
        elif period == "dayofyear":
            group = f"{self.dims.time}.dayofyear"
        elif period == "season":
            group = f"{self.dims.time}.season"
        else:
            raise ValueError(f"Unsupported period: {period!r}")

        counts = ds.groupby(group).count(dim=self.dims.time)

        if clim_ds is None:
            clim_ref = ds.groupby(group).mean(dim=self.dims.time, skipna=True)
        else:
            clim_ref = clim_ds

        clim_ref = clim_ref.where(counts >= min_periods)
        return ds.groupby(group) - clim_ref


    def clim_anom_corr(
        self,
        corr_reduce_dims: str | Sequence[str],
        period: ClimType = "month",
        min_periods: int = 2,
        eps: float = 1e-8,
    ) -> xr.Dataset:
        """
        Compute anomaly correlation relative to climatology.

        Anomalies are computed for both truth and models using either
        provided climatologies or internally derived ones, then correlated.

        Parameters
        ----------
        corr_reduce_dims : str or sequence of str
            Dimension(s) over which to compute correlation.
        period : {"month", "dayofyear", "season"}, optional
            Temporal grouping used for climatology.
        min_periods : int, optional
            Minimum number of samples required per group and correlation.
        eps : float, optional
            Threshold to avoid division by near-zero standard deviation.

        Returns
        -------
        xr.Dataset
            Dataset of anomaly correlations with a 'model' dimension.

        Notes
        -----
        `clim_data` must contain one climatology for truth and one per model.
        """
        corr_reduce_dims = self._as_tuple(corr_reduce_dims)

        if self.clim_data is None or len(self.clim_data) != len(self.model_data) + 1:
            raise ValueError(
                "acc_clim requires clim_data with length len(model_data) + 1 "
                "(truth climatology first, then one climatology per model)"
            )

        truth_anom = self._climatology_anom(
            self.truth_data,
            clim_ds=self.clim_data[0],
            period=period,
            min_periods=min_periods,
        )

        per_model = []

        for i, (model_ds, model_name) in enumerate(zip(self.model_data, self.model_names)):
            model_clim = self.clim_data[i + 1]

            model_anom = self._climatology_anom(
                model_ds,
                clim_ds=model_clim,
                period=period,
                min_periods=min_periods,
            )

            common_vars = [v for v in model_anom.data_vars if v in truth_anom.data_vars]
            if not common_vars:
                raise ValueError(
                    f"No common variables between truth and model {model_name!r}"
                )

            out_vars = {}
            for var in common_vars:
                out_vars[var] = self._corr_da(
                    model_anom[var],
                    truth_anom[var],
                    corr_reduce_dims=corr_reduce_dims,
                    min_periods=min_periods,
                    eps=eps,
                )

            per_model.append(
                xr.Dataset(out_vars).expand_dims(model=[model_name])
            )

        return xr.concat(per_model, dim="model")


    def clim_anom_corr_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        corr_reduce_dims: str | Sequence[str],
        period: ClimType = "month",
        min_periods: int = 2,
        eps: float = 1e-8,
    ) -> xr.Dataset:
        """
        Compute climatology-anomaly correlation between mean fields of model and truth.

        The data are first averaged over `data_mean_dims`, anomalies are then computed
        relative to climatology, and correlation is evaluated over `corr_reduce_dims`.

        Parameters
        ----------
        data_mean_dims : str or sequence of str
            Dimension(s) over which to compute the mean fields.
        corr_reduce_dims : str or sequence of str
            Dimension(s) over which to compute correlation.
        period : {"month", "dayofyear", "season"}, optional
            Temporal grouping used for climatology.
        min_periods : int, optional
            Minimum number of samples required per group and correlation.
        eps : float, optional
            Threshold to avoid division by near-zero standard deviation.

        Returns
        -------
        xr.Dataset
            Dataset of anomaly correlations with a 'model' dimension.

        Notes
        -----
        `clim_data` must contain one climatology for truth and one per model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        corr_reduce_dims = self._as_tuple(corr_reduce_dims)

        self._check_dims_overlap(
            data_mean_dims, corr_reduce_dims, "data_mean_dims", "corr_reduce_dims"
        )

        if self.clim_data is None or len(self.clim_data) != len(self.model_data) + 1:
            raise ValueError(
                "clim_anom_corr_of_mean requires clim_data with length len(model_data) + 1 "
                "(truth climatology first, then one climatology per model)"
            )

        truth_mean = self.truth_data.earthml.geo_mean(data_mean_dims)
        truth_clim = self.clim_data[0]
        if truth_clim is not None:
            truth_clim = truth_clim.earthml.geo_mean(data_mean_dims)

        truth_anom = self._climatology_anom(
            truth_mean,
            clim_ds=truth_clim,
            period=period,
            min_periods=min_periods,
        )

        per_model = []

        for i, (model_ds, model_name) in enumerate(zip(self.model_data, self.model_names)):
            model_mean = model_ds.earthml.geo_mean(data_mean_dims)

            model_clim = self.clim_data[i + 1]
            if model_clim is not None:
                model_clim = model_clim.earthml.geo_mean(data_mean_dims)

            model_anom = self._climatology_anom(
                model_mean,
                clim_ds=model_clim,
                period=period,
                min_periods=min_periods,
            )

            common_vars = [v for v in model_anom.data_vars if v in truth_anom.data_vars]
            if not common_vars:
                raise ValueError(
                    f"No common variables between truth and model {model_name!r}"
                )

            out_vars = {}
            for var in common_vars:
                out_vars[var] = self._corr_da(
                    model_anom[var],
                    truth_anom[var],
                    corr_reduce_dims=corr_reduce_dims,
                    min_periods=min_periods,
                    eps=eps,
                )

            per_model.append(
                xr.Dataset(out_vars).expand_dims(model=[model_name])
            )

        return xr.concat(per_model, dim="model")


    def spatial_anom_corr(
        self,
        corr_reduce_dims: str | Sequence[str],
        min_periods: int = 2,
        eps: float = 1e-8,
    ) -> xr.Dataset:
        """
        Compute correlation of spatial anomalies between models and truth.

        Spatial anomalies are defined by removing the latitude–longitude mean
        at each remaining coordinate (e.g., time, realization), then correlated.

        Parameters
        ----------
        corr_reduce_dims : str or sequence of str
            Dimension(s) over which to compute correlation.
        min_periods : int, optional
            Minimum number of valid paired samples required.
        eps : float, optional
            Threshold to avoid division by near-zero standard deviation.

        Returns
        -------
        xr.Dataset
            Dataset of correlation values with a 'model' dimension.

        Notes
        -----
        Spatial mean is computed using geo-weighted averaging.
        """
        corr_reduce_dims = self._as_tuple(corr_reduce_dims)

        truth_anom = self.truth_data - self.truth_data.earthml.geo_mean((self.dims.latitude, self.dims.longitude))

        per_model = []

        for model_ds, model_name in zip(self.model_data, self.model_names):

            model_anom = model_ds - model_ds.earthml.geo_mean((self.dims.latitude, self.dims.longitude))

            common_vars = [v for v in model_anom.data_vars if v in truth_anom.data_vars]
            if not common_vars:
                raise ValueError(
                    f"No common variables between truth and model {model_name!r}"
                )

            out_vars = {}
            for var in common_vars:
                out_vars[var] = self._corr_da(
                    model_anom[var],
                    truth_anom[var],
                    corr_reduce_dims=corr_reduce_dims,
                    min_periods=min_periods,
                    eps=eps,
                )

            per_model.append(
                xr.Dataset(out_vars).expand_dims(model=[model_name])
            )

        return xr.concat(per_model, dim="model")


    def spatial_anom_corr_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        corr_reduce_dims: str | Sequence[str],
        min_periods: int = 2,
        eps: float = 1e-8,
    ) -> xr.Dataset:
        """
        Compute spatial-anomaly correlation between mean fields of model and truth.

        The data are first averaged over `data_mean_dims`, spatial anomalies are then
        computed by removing the latitude-longitude mean, and correlation is evaluated
        over `corr_reduce_dims`.

        Parameters
        ----------
        data_mean_dims : str or sequence of str
            Dimension(s) over which to compute the mean fields.
        corr_reduce_dims : str or sequence of str
            Dimension(s) over which to compute correlation.
        min_periods : int, optional
            Minimum number of valid paired samples required.
        eps : float, optional
            Threshold to avoid division by near-zero standard deviation.

        Returns
        -------
        xr.Dataset
            Dataset of correlation values with a 'model' dimension.

        Notes
        -----
        Spatial mean is computed using geo-weighted averaging.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        corr_reduce_dims = self._as_tuple(corr_reduce_dims)

        self._check_dims_overlap(
            data_mean_dims, corr_reduce_dims, "data_mean_dims", "corr_reduce_dims"
        )

        spatial_dims = (self.dims.latitude, self.dims.longitude)

        truth_mean = self.truth_data.earthml.geo_mean(data_mean_dims)
        truth_anom = truth_mean - truth_mean.earthml.geo_mean(spatial_dims)

        per_model = []

        for model_ds, model_name in zip(self.model_data, self.model_names):
            model_mean = model_ds.earthml.geo_mean(data_mean_dims)
            model_anom = model_mean - model_mean.earthml.geo_mean(spatial_dims)

            common_vars = [v for v in model_anom.data_vars if v in truth_anom.data_vars]
            if not common_vars:
                raise ValueError(
                    f"No common variables between truth and model {model_name!r}"
                )

            out_vars = {}
            for var in common_vars:
                out_vars[var] = self._corr_da(
                    model_anom[var],
                    truth_anom[var],
                    corr_reduce_dims=corr_reduce_dims,
                    min_periods=min_periods,
                    eps=eps,
                )

            per_model.append(
                xr.Dataset(out_vars).expand_dims(model=[model_name])
            )

        return xr.concat(per_model, dim="model")
