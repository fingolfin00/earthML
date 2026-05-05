from typing import Sequence, Callable
import xarray as xr

from .dataclass import MetricData


class BaseMetrics:
    def __init__(
        self,
        truth_data: xr.Dataset,
        model_data: xr.Dataset | list[xr.Dataset],
        truth_name: str,
        model_names: str | list[str],
        clim_data: xr.Dataset | list[xr.Dataset] | None = None,
        mask_data: xr.Dataset | None = None,
    ):
        # Preprocess
        md = MetricData.from_inputs(
            truth_data=truth_data,
            model_data=model_data,
            truth_name=truth_name,
            model_names=model_names,
            clim_data=clim_data,
            mask_data=mask_data,
        )

        self._data = md

        # Unpack for convenience
        self.truth_data = md.truth_data
        self.model_data = md.model_data
        self.clim_data = md.clim_data
        self.mask_data = md.mask_data
        self.truth_name = md.truth_name
        self.model_names = md.model_names
        self.dims = md.dims

    @staticmethod
    def _as_tuple(dims: str | Sequence[str]) -> tuple[str, ...]:
        return (dims,) if isinstance(dims, str) else tuple(dims)
    
    @staticmethod
    def _check_dims_overlap(
        dims1: tuple[str, ...],
        dims2: tuple[str, ...],
        name1: str = "dims1",
        name2: str = "dims2",
    ) -> None:
        overlap = set(dims1) & set(dims2)
        if overlap:
            raise ValueError(
                f"Dimensions cannot appear in both {name1} and {name2}: {sorted(overlap)}"
            )

    def _common_vars(self, model_ds: xr.Dataset, model_name: str) -> list[str]:
        common_vars = [v for v in model_ds.data_vars if v in self.truth_data.data_vars]
        if not common_vars:
            raise ValueError(f"No common variables between truth and model {model_name!r}")
        return common_vars

    def _apply_metric_per_model(
        self,
        metric_func: Callable[[xr.DataArray, xr.DataArray], xr.DataArray],
    ) -> xr.Dataset:
        """
        Loop over models and apply the metric_func (signature [xr.DataArray, xr.DataArray] -> xr.DataArray).
        Then concatenate metrics per model in a new xr.Dataset with the new dimension "model".
        """
        per_model = []

        for model_ds, model_name in zip(self.model_data, self.model_names):
            out_vars = {}

            for var in self._common_vars(model_ds, model_name):
                out_vars[var] = metric_func(model_ds[var], self.truth_data[var])

            per_model.append(
                xr.Dataset(out_vars).expand_dims(model=[model_name])
            )

        return xr.concat(per_model, dim="model")
    