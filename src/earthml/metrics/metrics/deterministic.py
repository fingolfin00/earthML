from typing import Literal, Sequence
import numpy as np
import xarray as xr

from .base import BaseMetrics


NormType = Literal["mean", "std", "range"]
ClimType = Literal["month", "dayofyear", "season"]


class DeterministicMetrics(BaseMetrics):
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


    def _climatology_error(
        self,
        truth: xr.DataArray,
        clim: xr.DataArray | None,
        period: ClimType,
    ) -> xr.DataArray:
        if period == "month":
            group = f"{self.dims.time}.month"
        elif period == "dayofyear":
            group = f"{self.dims.time}.dayofyear"
        elif period == "season":
            group = f"{self.dims.time}.season"
        else:
            raise ValueError(f"Unsupported period: {period!r}")

        clim_ref = truth.groupby(group).mean(dim=self.dims.time, skipna=True) if clim is None else clim
        return truth.groupby(group) - clim_ref


    def _normalization_factor(
        self,
        truth: xr.DataArray,
        dims: tuple[str, ...],
        norm: NormType,
    ) -> xr.DataArray:
        if norm == "mean":
            factor = np.abs(truth).earthml.geo_mean(dims)
        elif norm == "std":
            factor = truth.earthml.geo_std(dims)
        elif norm == "range":
            factor = truth.max(dim=dims) - truth.min(dim=dims)
        else:
            raise ValueError(f"Unsupported norm={norm!r}. Choose from 'mean', 'std', 'range'.")

        factor = factor.where(factor != 0)
        return factor


    def error(self) -> xr.Dataset:
        """
        Compute the raw signed error field (model minus truth).

        Returns
        -------
        xr.Dataset
            Signed error for each model with the original data dimensions preserved.
        """
        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            return model - truth

        return self._apply_metric_per_model(metric_func)


    def error_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute the signed error of mean fields (model minus truth).

        The data are first averaged over `data_mean_dims`, then the signed error
        is computed without any additional reduction.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.

        Returns
        -------
        xr.Dataset
            Signed error of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)
            return model_mean - truth_mean

        return self._apply_metric_per_model(metric_func)


    def rmse(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute root mean squared error (RMSE) between model and truth.

        The squared error is averaged over `metric_mean_dims` using a geographic mean,
        then square-rooted.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the squared error.

        Returns
        -------
        xr.Dataset
            RMSE for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            sq_err = (model - truth) ** 2
            mse = sq_err.earthml.geo_mean(metric_mean_dims)
            return np.sqrt(mse)

        return self._apply_metric_per_model(metric_func)


    def rmse_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute RMSE of mean fields between model and truth.

        The data are first averaged over `data_mean_dims`, then RMSE is computed
        over `metric_mean_dims`.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the squared error.

        Returns
        -------
        xr.Dataset
            RMSE of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)
            sq_err = (model_mean - truth_mean) ** 2
            mse = sq_err.earthml.geo_mean(metric_mean_dims)
            return np.sqrt(mse)

        return self._apply_metric_per_model(metric_func)


    def mae(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute mean absolute error (MAE) between model and truth.

        The absolute error is averaged over `metric_mean_dims` using a geographic mean.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the absolute error.

        Returns
        -------
        xr.Dataset
            MAE for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            abs_err = np.abs(model - truth)
            return abs_err.earthml.geo_mean(metric_mean_dims)

        return self._apply_metric_per_model(metric_func)


    def mae_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute MAE of mean fields between model and truth.

        The data are first averaged over `data_mean_dims`, then MAE is computed
        over `metric_mean_dims`.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the absolute error.

        Returns
        -------
        xr.Dataset
            MAE of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)
            abs_err = np.abs(model_mean - truth_mean)
            return abs_err.earthml.geo_mean(metric_mean_dims)

        return self._apply_metric_per_model(metric_func)
    
    def crmse(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute centered RMSE (CRMSE) between model and truth.

        The mean over `metric_mean_dims` is removed from both model and truth
        before computing RMSE, isolating pattern errors.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to compute and remove the mean, and to average the error.

        Returns
        -------
        xr.Dataset
            CRMSE for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_anom = model - model.earthml.geo_mean(metric_mean_dims)
            truth_anom = truth - truth.earthml.geo_mean(metric_mean_dims)
            sq_err = (model_anom - truth_anom) ** 2
            mse = sq_err.earthml.geo_mean(metric_mean_dims)
            return np.sqrt(mse)

        return self._apply_metric_per_model(metric_func)


    def crmse_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute CRMSE of mean fields between model and truth.

        The data are first averaged over `data_mean_dims`. The mean over
        `metric_mean_dims` is then removed before computing RMSE.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to remove the mean and average the error.

        Returns
        -------
        xr.Dataset
            CRMSE of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)

            model_anom = model_mean - model_mean.earthml.geo_mean(metric_mean_dims)
            truth_anom = truth_mean - truth_mean.earthml.geo_mean(metric_mean_dims)

            sq_err = (model_anom - truth_anom) ** 2
            mse = sq_err.earthml.geo_mean(metric_mean_dims)
            return np.sqrt(mse)

        return self._apply_metric_per_model(metric_func)


    def bias(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute mean bias (model minus truth).

        The error is averaged over `metric_mean_dims` using a geographic mean.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the error.

        Returns
        -------
        xr.Dataset
            Bias for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            err = model - truth
            return err.earthml.geo_mean(metric_mean_dims)

        return self._apply_metric_per_model(metric_func)


    def bias_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute bias of mean fields (model minus truth).

        The data are first averaged over `data_mean_dims`, then bias is computed
        over `metric_mean_dims`.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the error.

        Returns
        -------
        xr.Dataset
            Bias of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)
            err = model_mean - truth_mean
            return err.earthml.geo_mean(metric_mean_dims)

        return self._apply_metric_per_model(metric_func)


    def error_std(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute the standard deviation of the signed error (model minus truth).

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to compute the error standard deviation.

        Returns
        -------
        xr.Dataset
            Error standard deviation for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            err = model - truth
            return err.earthml.geo_std(metric_mean_dims)

        return self._apply_metric_per_model(metric_func)


    def error_std_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute the standard deviation of the signed error of mean fields.

        The data are first averaged over `data_mean_dims`, then the error
        standard deviation is computed over `metric_mean_dims`.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to compute the error standard deviation.

        Returns
        -------
        xr.Dataset
            Error standard deviation of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)
            err = model_mean - truth_mean
            return err.earthml.geo_std(metric_mean_dims)

        return self._apply_metric_per_model(metric_func)


    def variance_ratio(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute the ratio between model and truth standard deviation.

        Values below 1 indicate under-variability, while values above 1 indicate
        over-variability relative to the truth.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to compute the standard deviations.

        Returns
        -------
        xr.Dataset
            Ratio of model standard deviation to truth standard deviation.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_std = model.earthml.geo_std(metric_mean_dims)
            truth_std = truth.earthml.geo_std(metric_mean_dims)
            truth_std = truth_std.where(truth_std != 0)
            return model_std / truth_std

        return self._apply_metric_per_model(metric_func)


    def variance_ratio_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute the ratio between model and truth standard deviation of mean fields.

        The data are first averaged over `data_mean_dims`, then the standard
        deviation ratio is computed over `metric_mean_dims`.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to compute the standard deviations.

        Returns
        -------
        xr.Dataset
            Variance ratio of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)
            model_std = model_mean.earthml.geo_std(metric_mean_dims)
            truth_std = truth_mean.earthml.geo_std(metric_mean_dims)
            truth_std = truth_std.where(truth_std != 0)
            return model_std / truth_std

        return self._apply_metric_per_model(metric_func)


    def nrmse(
        self,
        metric_mean_dims: str | Sequence[str],
        norm: NormType = "std",
    ) -> xr.Dataset:
        """
        Compute normalized RMSE (NRMSE) between model and truth.

        RMSE is divided by a normalization factor derived from the truth over
        `metric_mean_dims` (mean, standard deviation, or range).

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the squared error.
        norm : {"mean", "std", "range"}, default "std"
            Type of normalization applied to the RMSE.

        Returns
        -------
        xr.Dataset
            Normalized RMSE for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            sq_err = (model - truth) ** 2
            mse = sq_err.earthml.geo_mean(metric_mean_dims)
            rmse = np.sqrt(mse)
            factor = self._normalization_factor(truth, metric_mean_dims, norm)
            return rmse / factor

        return self._apply_metric_per_model(metric_func)


    def nrmse_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
        norm: NormType = "std",
    ) -> xr.Dataset:
        """
        Compute normalized RMSE of mean fields.

        The data are first averaged over `data_mean_dims`, then RMSE is computed
        and normalized using statistics of the truth mean.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the squared error.
        norm : {"mean", "std", "range"}, default "std"
            Type of normalization applied to the RMSE.

        Returns
        -------
        xr.Dataset
            Normalized RMSE of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)

            sq_err = (model_mean - truth_mean) ** 2
            mse = sq_err.earthml.geo_mean(metric_mean_dims)
            rmse = np.sqrt(mse)

            factor = self._normalization_factor(truth_mean, metric_mean_dims, norm)
            return rmse / factor

        return self._apply_metric_per_model(metric_func)


    def nmae(
        self,
        metric_mean_dims: str | Sequence[str],
        norm: NormType = "std",
    ) -> xr.Dataset:
        """
        Compute normalized MAE (NMAE) between model and truth.

        MAE is divided by a normalization factor derived from the truth over
        `metric_mean_dims`.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the absolute error.
        norm : {"mean", "std", "range"}, default "std"
            Type of normalization applied to the MAE.

        Returns
        -------
        xr.Dataset
            Normalized MAE for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            abs_err = np.abs(model - truth)
            mae = abs_err.earthml.geo_mean(metric_mean_dims)
            factor = self._normalization_factor(truth, metric_mean_dims, norm)
            return mae / factor

        return self._apply_metric_per_model(metric_func)


    def nmae_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
        norm: NormType = "std",
    ) -> xr.Dataset:
        """
        Compute normalized MAE of mean fields.

        The data are first averaged over `data_mean_dims`, then MAE is computed
        and normalized using statistics of the truth mean.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the absolute error.
        norm : {"mean", "std", "range"}, default "std"
            Type of normalization applied to the MAE.

        Returns
        -------
        xr.Dataset
            Normalized MAE of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)

            abs_err = np.abs(model_mean - truth_mean)
            mae = abs_err.earthml.geo_mean(metric_mean_dims)

            factor = self._normalization_factor(truth_mean, metric_mean_dims, norm)
            return mae / factor

        return self._apply_metric_per_model(metric_func)


    def nbias(
        self,
        metric_mean_dims: str | Sequence[str],
        norm: NormType = "std",
    ) -> xr.Dataset:
        """
        Compute normalized bias (model minus truth).

        Bias is divided by a normalization factor derived from the truth over
        `metric_mean_dims`.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the error.
        norm : {"mean", "std", "range"}, default "std"
            Type of normalization applied to the bias.

        Returns
        -------
        xr.Dataset
            Normalized bias for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            err = model - truth
            bias = err.earthml.geo_mean(metric_mean_dims)
            factor = self._normalization_factor(truth, metric_mean_dims, norm)
            return bias / factor

        return self._apply_metric_per_model(metric_func)


    def nbias_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
        norm: NormType = "std",
    ) -> xr.Dataset:
        """
        Compute normalized bias of mean fields.

        The data are first averaged over `data_mean_dims`, then bias is computed
        and normalized using statistics of the truth mean.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the error.
        norm : {"mean", "std", "range"}, default "std"
            Type of normalization applied to the bias.

        Returns
        -------
        xr.Dataset
            Normalized bias of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)

            err = model_mean - truth_mean
            bias = err.earthml.geo_mean(metric_mean_dims)

            factor = self._normalization_factor(truth_mean, metric_mean_dims, norm)
            return bias / factor

        return self._apply_metric_per_model(metric_func)


    def ncrmse(
        self,
        metric_mean_dims: str | Sequence[str],
        norm: NormType = "std",
    ) -> xr.Dataset:
        """
        Compute normalized centered RMSE (NCRMSE).

        CRMSE is computed after removing the mean over `metric_mean_dims`, then
        normalized by a factor derived from the truth.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to remove the mean and average the error.
        norm : {"mean", "std", "range"}, default "std"
            Type of normalization applied to the CRMSE.

        Returns
        -------
        xr.Dataset
            Normalized CRMSE for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_anom = model - model.earthml.geo_mean(metric_mean_dims)
            truth_anom = truth - truth.earthml.geo_mean(metric_mean_dims)
            sq_err = (model_anom - truth_anom) ** 2
            mse = sq_err.earthml.geo_mean(metric_mean_dims)
            crmse = np.sqrt(mse)
            factor = self._normalization_factor(truth, metric_mean_dims, norm)
            return crmse / factor

        return self._apply_metric_per_model(metric_func)


    def ncrmse_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
        norm: NormType = "std",
    ) -> xr.Dataset:
        """
        Compute normalized CRMSE of mean fields.

        The data are first averaged over `data_mean_dims`. CRMSE is then computed
        (after removing the mean over `metric_mean_dims`) and normalized.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to remove the mean and average the error.
        norm : {"mean", "std", "range"}, default "std"
            Type of normalization applied to the CRMSE.

        Returns
        -------
        xr.Dataset
            Normalized CRMSE of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)

            model_anom = model_mean - model_mean.earthml.geo_mean(metric_mean_dims)
            truth_anom = truth_mean - truth_mean.earthml.geo_mean(metric_mean_dims)

            sq_err = (model_anom - truth_anom) ** 2
            mse = sq_err.earthml.geo_mean(metric_mean_dims)
            crmse = np.sqrt(mse)

            factor = self._normalization_factor(truth_mean, metric_mean_dims, norm)
            return crmse / factor

        return self._apply_metric_per_model(metric_func)


    def r2(
        self,
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute coefficient of determination (R²) between model and truth.

        R² is defined as 1 - MSE / Var(truth), where both MSE and truth variance
        are computed over `metric_mean_dims` using geographic reductions.

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to compute the metric.

        Returns
        -------
        xr.Dataset
            R² for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            mse = ((model - truth) ** 2).earthml.geo_mean(metric_mean_dims)
            var_truth = truth.earthml.geo_std(metric_mean_dims) ** 2
            var_truth = var_truth.where(var_truth != 0)
            return 1.0 - mse / var_truth

        return self._apply_metric_per_model(metric_func)


    def r2_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
    ) -> xr.Dataset:
        """
        Compute coefficient of determination (R²) for mean fields.

        The data are first averaged over `data_mean_dims`, then R² is computed as
        1 - MSE / Var(truth_mean) over `metric_mean_dims`.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to compute the metric.

        Returns
        -------
        xr.Dataset
            R² of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(
            data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims"
        )

        def metric_func(model: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
            model_mean = model.earthml.geo_mean(data_mean_dims)
            truth_mean = truth.earthml.geo_mean(data_mean_dims)

            mse = ((model_mean - truth_mean) ** 2).earthml.geo_mean(metric_mean_dims)
            var_truth = truth_mean.earthml.geo_std(metric_mean_dims) ** 2
            var_truth = var_truth.where(var_truth != 0)
            return 1.0 - mse / var_truth

        return self._apply_metric_per_model(metric_func)


    def rmse_skill_clim(
        self,
        metric_mean_dims: str | Sequence[str],
        period: ClimType = "month",
    ) -> xr.Dataset:
        """
        Compute RMSE skill score relative to climatology.

        Skill = 1 - RMSE_model / RMSE_climatology

        Parameters
        ----------
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the squared error.
        period : {"month", "dayofyear", "season"}, default "month"
            Temporal grouping used for climatology.

        Returns
        -------
        xr.Dataset
            RMSE skill score for each model.
        """
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        per_model = []

        for i, (model_ds, model_name) in enumerate(zip(self.model_data, self.model_names)):
            clim_ds = self.clim_data[i + 1] if self.clim_data is not None else None
            out_vars = {}

            for var in self._common_vars(model_ds, model_name):
                model = model_ds[var]
                truth = self.truth_data[var]
                clim = None if clim_ds is None else clim_ds[var]

                rmse_model = np.sqrt(((model - truth) ** 2).earthml.geo_mean(metric_mean_dims))
                rmse_clim = np.sqrt((self._climatology_error(truth, clim, period) ** 2).earthml.geo_mean(metric_mean_dims))
                rmse_clim = rmse_clim.where(rmse_clim != 0)

                out_vars[var] = 1.0 - rmse_model / rmse_clim

            per_model.append(xr.Dataset(out_vars).expand_dims(model=[model_name]))

        return xr.concat(per_model, dim="model")


    def rmse_skill_clim_of_mean(
        self,
        data_mean_dims: str | Sequence[str],
        metric_mean_dims: str | Sequence[str],
        period: ClimType = "month",
    ) -> xr.Dataset:
        """
        Compute RMSE skill score of mean fields relative to climatology.

        The data are first averaged over `data_mean_dims`, then the skill score
        is computed as 1 - RMSE_model_mean / RMSE_climatology_mean.

        Parameters
        ----------
        data_mean_dims : str or Sequence[str]
            Dimensions over which to compute the mean fields.
        metric_mean_dims : str or Sequence[str]
            Dimensions over which to average the squared error.
        period : {"month", "dayofyear", "season"}, default "month"
            Temporal grouping used for climatology.

        Returns
        -------
        xr.Dataset
            RMSE skill score of mean fields for each model.
        """
        data_mean_dims = self._as_tuple(data_mean_dims)
        metric_mean_dims = self._as_tuple(metric_mean_dims)

        self._check_dims_overlap(data_mean_dims, metric_mean_dims, "data_mean_dims", "metric_mean_dims")

        per_model = []

        for i, (model_ds, model_name) in enumerate(zip(self.model_data, self.model_names)):
            clim_ds = self.clim_data[i + 1] if self.clim_data is not None else None
            out_vars = {}

            for var in self._common_vars(model_ds, model_name):
                model_mean = model_ds[var].earthml.geo_mean(data_mean_dims)
                truth_mean = self.truth_data[var].earthml.geo_mean(data_mean_dims)

                clim = None
                if clim_ds is not None:
                    clim = clim_ds[var]
                    clim_mean_dims = tuple(dim for dim in data_mean_dims if dim in clim.dims)
                    if clim_mean_dims:
                        clim = clim.earthml.geo_mean(clim_mean_dims)

                rmse_model = np.sqrt(((model_mean - truth_mean) ** 2).earthml.geo_mean(metric_mean_dims))
                rmse_clim = np.sqrt((self._climatology_error(truth_mean, clim, period) ** 2).earthml.geo_mean(metric_mean_dims))
                rmse_clim = rmse_clim.where(rmse_clim != 0)

                out_vars[var] = 1.0 - rmse_model / rmse_clim

            per_model.append(xr.Dataset(out_vars).expand_dims(model=[model_name]))

        return xr.concat(per_model, dim="model")
