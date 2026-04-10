from typing import Callable

from ..logging import get_logger


logger = get_logger(__name__)


class EarthMLConvert:
    def convert_unit(self, convert_unit_d: dict[Callable, str]):
        # TODO add a description
        ds = self._obj

        found_candidate = False
        for var_name, (func, target_unit) in convert_unit_d.items():
            if var_name not in ds.data_vars:
                logger.warning(
                    "Exact match variable %s not found in dataset for unit conversion. Try matching within available variables %s",
                    var_name,
                    list(ds.data_vars.keys()),
                )
                for var_candidate in ds.data_vars.keys():
                    if var_name.lower() in var_candidate.lower() or var_candidate.lower() in var_name.lower():
                        logger.info("   Found candidate variable %s for conversion of %s", var_candidate, var_name)
                        var_name = var_candidate
                        found_candidate = True
                        break
                if not found_candidate:
                    logger.warning("No variable found for conversion of %s, skipping...", var_name)
                    continue

            da = ds[var_name]
            src_unit = da.attrs.get("units", None)

            # Skip if already in target unit
            if src_unit == target_unit:
                logger.info("Variable %s already in target unit %s, skipping conversion.", var_name, target_unit)
                continue

            logger.info(
                "Converting unit of variable %s%s to %s",
                var_name,
                "" if src_unit is None else f" from {src_unit}",
                target_unit,
            )

            # Preserve metadata
            old_attrs = dict(da.attrs)
            old_encoding = getattr(da, "encoding", {}).copy()

            out = func(da)
            if not hasattr(out, "dims"):
                raise TypeError(
                    f"Unit conversion for {var_name} must return an xarray.DataArray "
                    f"(got {type(out)!r})."
                )

            # Restore attrs, overwrite units
            out.attrs = old_attrs
            out.attrs["units"] = target_unit

            # Optional: preserve encoding (useful for NetCDF writing)
            if hasattr(out, "encoding"):
                out.encoding = old_encoding

            ds[var_name] = out

        return ds
