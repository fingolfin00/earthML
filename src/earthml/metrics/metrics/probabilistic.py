from typing import Sequence
import numpy as np
import xarray as xr

from .base import BaseMetrics


class ProbabilisticMetrics(BaseMetrics):
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
            mask_data=mask_data,
        )

        if self.dims.realization is None:
            raise ValueError(
                "ProbabilisticMetrics requires a realization dimension in model_data."
            )


    def _ensemble_spread_da(
        self,
        model: xr.DataArray,
    ) -> xr.DataArray:
        realization_dim = self.dims.realization

        if realization_dim not in model.dims:
            raise ValueError(
                f"Model data must contain realization dimension {realization_dim!r}."
            )

        return model.std(dim=realization_dim, skipna=True)


    def _crps_ensemble_da(
        self,
        model: xr.DataArray,
        truth: xr.DataArray,
    ) -> xr.DataArray:
        """
        Compute pointwise CRPS for an ensemble forecast and observation.

        Parameters
        ----------
        model : xr.DataArray
            Ensemble forecast with realization dimension.
        truth : xr.DataArray
            Truth/observation with or without realization dimension.
            If realization is present, it's averaged out

        Returns
        -------
        xr.DataArray
            Pointwise CRPS with the realization dimension removed.
        """
        realization_dim = self.dims.realization

        if realization_dim not in model.dims:
            raise ValueError(
                f"Model data must contain realization dimension {realization_dim!r}."
            )
        if realization_dim in truth.dims:
            truth = truth.mean(realization_dim)

        # First term: mean absolute error between members and truth
        term1 = np.abs(model - truth).mean(dim=realization_dim, skipna=True)

        # Second term: half the mean absolute difference between ensemble members
        m1 = model.rename({realization_dim: f"{realization_dim}_1"})
        m2 = model.rename({realization_dim: f"{realization_dim}_2"})
        term2 = 0.5 * np.abs(m1 - m2).mean(
            dim=(f"{realization_dim}_1", f"{realization_dim}_2"),
            skipna=True,
        )

        return term1 - term2


    def crps(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute continuous ranked probability score (CRPS) for ensemble forecasts.

        CRPS is computed pointwise from the ensemble forecast and truth, then
        averaged over `metric_mean_dims` using a geographic mean.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the pointwise CRPS.

        Returns
        -------
        xr.Dataset
            CRPS for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            crps = self._crps_ensemble_da(model, truth)
            return crps.earthml.geo_mean(metric_mean_dims)

        return self._apply_metric_per_model(metric_func)


    def crps_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute CRPS for ensemble forecasts of mean fields.

        Forecast and truth are first averaged over `data_mean_dims`, then CRPS
        is computed and averaged over `metric_mean_dims`.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the pointwise CRPS.

        Returns
        -------
        xr.Dataset
            CRPS of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(
            data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims"
        )

        realization_dim = self.dims.realization
        mean_dims_model = tuple(d for d in data_mean_dims if d != realization_dim)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(mean_dims_model) if mean_dims_model else model
            truth_mean = truth.earthml.geo_mean(data_mean_dims)

            crps = self._crps_ensemble_da(model_mean, truth_mean)
            return crps.earthml.geo_mean(metric_mean_dims)

        return self._apply_metric_per_model(metric_func)


    def crps_skill_clim(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute CRPS skill score relative to climatology.

        Skill = 1 - CRPS_model / CRPS_climatology

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average CRPS.

        Returns
        -------
        xr.Dataset
            CRPS skill score for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        if self.clim_data is None or len(self.clim_data) != len(self.model_data) + 1:
            raise ValueError(
                "crps_skill_clim requires clim_data with one entry for truth and one per model"
            )

        per_model = []

        for i, (model_ds, model_name) in enumerate(zip(self.model_data, self.model_names)):
            clim_ds = self.clim_data[i + 1]

            common_vars = [v for v in model_ds.data_vars if v in self.truth_data.data_vars]
            if not common_vars:
                raise ValueError(f"No common variables for model {model_name!r}")

            out_vars = {}

            for var in common_vars:
                model = model_ds[var]
                truth = self.truth_data[var]

                if clim_ds is None:
                    clim = truth.groupby(f"{self.dims.time}.month").mean(dim=self.dims.time, skipna=True)
                else:
                    clim = clim_ds[var]

                crps_model = self._crps_ensemble_da(model, truth)
                crps_model = crps_model.earthml.geo_mean(metric_mean_dims)

                # climatology treated as deterministic forecast
                crps_clim = np.abs(clim - truth)
                crps_clim = crps_clim.earthml.geo_mean(metric_mean_dims)

                crps_clim = crps_clim.where(crps_clim != 0)

                out_vars[var] = 1.0 - crps_model / crps_clim

            per_model.append(
                xr.Dataset(out_vars).expand_dims(model=[model_name])
            )

        return xr.concat(per_model, dim="model")


    def spread(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute ensemble spread as the mean member standard deviation.

        The ensemble standard deviation is computed pointwise over the
        realization dimension, then averaged over `metric_mean_dims`.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the pointwise spread.

        Returns
        -------
        xr.Dataset
            Ensemble spread for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            del truth
            spread = self._ensemble_spread_da(model)
            return spread.earthml.geo_mean(metric_mean_dims)

        return self._apply_metric_per_model(metric_func)


    def spread_error_ratio(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute the ratio between ensemble spread and ensemble-mean RMSE.

        A value near 1 suggests the ensemble spread is commensurate with the
        error of the ensemble mean, while values below or above 1 indicate
        under-dispersion or over-dispersion respectively.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average spread and error.

        Returns
        -------
        xr.Dataset
            Spread-to-error ratio for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)
        realization_dim = self.dims.realization

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            if realization_dim in truth.dims:
                truth = truth.mean(dim=realization_dim, skipna=True)
            spread = self._ensemble_spread_da(model).earthml.geo_mean(metric_mean_dims)
            ens_mean = model.mean(dim=realization_dim, skipna=True)
            rmse = np.sqrt(((ens_mean - truth) ** 2).earthml.geo_mean(metric_mean_dims))
            rmse = rmse.where(rmse != 0)
            return spread / rmse

        return self._apply_metric_per_model(metric_func)
