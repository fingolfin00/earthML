from datetime import datetime, timedelta
from datetime import time as datetime_time
from dateutil.relativedelta import relativedelta
from typing import Sequence

import numpy as np
import xarray as xr

from ...base import TimeRange


def _var_names(vars: str | Sequence[str]) -> str: # TODO remove?
    var_list = vars if isinstance(vars, Sequence) else [vars]
    return "".join(v for v in var_list)


def _floor_to_midnight(dt: datetime) -> datetime:
    return datetime.combine(dt.date(), datetime_time.min, tzinfo=dt.tzinfo)

def _month_start(dt):
    # assumes dt already midnight-aligned
    return dt.replace(day=1)

def _round_to_nearest_month_start(dt):
    """
    Round dt to the nearest month start (00:00 on the 1st).
    Ties go to the earlier month start.
    """
    dt = _floor_to_midnight(dt)
    lo = _month_start(dt)
    hi = _month_start(dt + relativedelta(months=1))
    if (dt - lo) <= (hi - dt):
        return lo
    return hi

def half_train_periods_days(
    base: TimeRange,
    min_months: int = 3,
    anchor: str = "end",  # "end" or "start"
    month_start: bool = False,
) -> Sequence[TimeRange]:
    if base.end <= base.start:
        raise ValueError("base.end must be after base.start")
    if anchor not in {"end", "start"}:
        raise ValueError("anchor must be 'end' or 'start'")

    # Normalize endpoints to midnight to avoid hour drift
    start0 = _floor_to_midnight(base.start)
    end0   = _floor_to_midnight(base.end)

    # Minimum length in days, calendar months threshold
    if anchor == "end":
        min_start = end0 - relativedelta(months=min_months)
        min_days = (end0 - _floor_to_midnight(min_start)).days
    else:
        min_end = start0 + relativedelta(months=min_months)
        min_days = (_floor_to_midnight(min_end) - start0).days

    total_days = (end0 - start0).days
    if total_days <= 0:
        raise ValueError("After midnight alignment, range has no full days.")

    out: Sequence[TimeRange] = []
    days = total_days
    while days >= min_days:
        if anchor == "end":
            tr_start = end0 - timedelta(days=days)
            tr_end = end0

            if month_start:
                tr_start = _round_to_nearest_month_start(tr_start)
                # keep within base bounds
                if tr_start < start0:
                    tr_start = start0
        else:
            tr_start = start0
            tr_end = start0 + timedelta(days=days)

            if month_start:
                tr_end = _round_to_nearest_month_start(tr_end)
                # keep within base bounds
                if tr_end > end0:
                    tr_end = end0

        # Drop degenerate / inverted ranges that snapping could create
        if tr_end > tr_start:
            out.append(TimeRange(start=tr_start, end=tr_end, freq=base.freq, shifted=base.shifted))

        days //= 2  # day-granular

    return out

#TODO issue if train_period is only one month
def halved_windows_split_by_cutoff(
    base: TimeRange,
    cutoff_end: datetime,          # e.g. datetime(2014, 12, 31)
    min_months: int = 3,
    anchor: str = "end",           # "end" (default) or "start" for the halved window
    post_starts_next_day: bool = True,
    month_start: bool = False,
) -> Sequence[Sequence["TimeRange"]]:
    """
    Builds progressively halved windows (day-granular). For each window:
      - if cutoff_end lies inside the window (inclusive), returns [pre, post]
      - otherwise returns [window] only

    Output is a list of lists. Each inner list has length 1 or 2.
    """

    if anchor not in {"end", "start"}:
        raise ValueError("anchor must be 'end' or 'start'")

    # Day-align everything to avoid hour drift
    base_start = _floor_to_midnight(base.start)
    base_end   = _floor_to_midnight(base.end)
    cutoff0    = _floor_to_midnight(cutoff_end)

    if base_end <= base_start:
        raise ValueError("base.end must be after base.start")

    # Minimum window length in whole days (using calendar months)
    if anchor == "end":
        min_start = base_end - relativedelta(months=min_months)
        min_days = (base_end - _floor_to_midnight(min_start)).days
    else:
        min_end = base_start + relativedelta(months=min_months)
        min_days = (_floor_to_midnight(min_end) - base_start).days

    total_days = (base_end - base_start).days
    days = total_days

    out: Sequence[TimeRange] = []

    while days >= min_days and days > 0:
        # Build the halved window (day-only)
        if anchor == "end":
            win_start = base_end - timedelta(days=days)
            win_end = base_end

            if month_start:
                win_start = _round_to_nearest_month_start(win_start)
                # keep within base bounds
                if win_start < base_start:
                    win_start = base_start
        else:
            win_start = base_start
            win_end = base_start + timedelta(days=days)

            if month_start:
                win_end = _round_to_nearest_month_start(win_end)
                # keep within base bounds
                if win_end > base_end:
                    win_end = base_end

        window = TimeRange(start=win_start, end=win_end, freq=base.freq, shifted=base.shifted)

        # If cutoff is inside, split; else return just the window
        if win_start <= cutoff0 <= win_end:
            pre = TimeRange(start=win_start, end=cutoff0, freq=base.freq, shifted=base.shifted)

            post_start = cutoff0 + timedelta(days=1) if post_starts_next_day else cutoff0
            if post_start <= win_end:
                post = TimeRange(start=post_start, end=win_end, freq=base.freq, shifted=base.shifted)
                out.append([pre, post])
            else:
                # cutoff is at/near the end so post would be empty
                out.append([pre])
        else:
            out.append([window])

        days //= 2

    return out


def _as_mask_dataset(mask: xr.Dataset | xr.DataArray | None, *, name: str = "mask") -> xr.Dataset | None:
    if mask is None:
        return None
    if isinstance(mask, xr.DataArray):
        mask_name = mask.name or name
        return mask.to_dataset(name=mask_name)
    return mask


def coord_resolution(coord: xr.DataArray) -> float | None:
    values = np.asarray(coord.values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        return None
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
    if diffs.size == 0:
        return None
    return float(np.abs(diffs).mean())


def project_mask_to_reference_grid(
    mask_ds: xr.Dataset | xr.DataArray | None,
    reference_ds: xr.Dataset,
) -> xr.Dataset | None:
    mask_ds = _as_mask_dataset(mask_ds)
    if mask_ds is None:
        return None

    mask_ds = mask_ds.earthml.normalize_dims_and_coords()
    reference_ds = reference_ds.earthml.normalize_dims_and_coords()
    projected = mask_ds

    guessed_coords = reference_ds.earthml.guessed_coords
    for coord_name in (guessed_coords.latitude, guessed_coords.longitude):
        if coord_name is None or coord_name not in reference_ds.coords or coord_name not in projected.coords:
            continue

        ref_coord = reference_ds[coord_name]
        mask_coord = projected[coord_name]
        if ref_coord.ndim != 1 or mask_coord.ndim != 1:
            continue

        ref_res = coord_resolution(ref_coord)
        mask_res = coord_resolution(mask_coord)
        tol_candidates = [res for res in (ref_res, mask_res) if res is not None and np.isfinite(res)]
        tolerance = 0.51 * max(tol_candidates) if tol_candidates else None

        if tolerance is None:
            projected = projected.reindex({coord_name: ref_coord})
        else:
            projected = projected.reindex(
                {coord_name: ref_coord},
                method="nearest",
                tolerance=tolerance,
            )

    return projected


def select_mask_for_indexers(
    mask_ds: xr.Dataset | xr.DataArray | None,
    indexers: dict[str, object],
) -> xr.Dataset | None:
    mask_ds = _as_mask_dataset(mask_ds)
    if mask_ds is None:
        return None

    mask_indexers = (
        {
            dim: value
            for dim, value in indexers.items()
            if dim in mask_ds.dims or dim in mask_ds.coords
        }
        if indexers else {}
    )
    selected = mask_ds.sel(mask_indexers, drop=True) if mask_indexers else mask_ds
    if selected is not None and selected.data_vars and not bool(selected.to_array().notnull().any()):
        return None
    return selected


def combine_masks(
    saved_mask: xr.Dataset | xr.DataArray | None,
    external_mask: xr.Dataset | xr.DataArray | None,
    *,
    output_name: str = "mask",
) -> xr.Dataset | None:
    saved_mask = _as_mask_dataset(saved_mask, name=output_name)
    external_mask = _as_mask_dataset(external_mask, name=output_name)

    if saved_mask is None:
        return external_mask
    if external_mask is None:
        return saved_mask

    saved_aligned, external_aligned = xr.align(saved_mask, external_mask, join="inner")
    saved_valid = saved_aligned.to_array().all("variable")
    external_valid = external_aligned.to_array().all("variable")
    combined_valid = saved_valid & external_valid
    return combined_valid.to_dataset(name=output_name)


def apply_mask_to_dataset(
    dataset: xr.Dataset,
    mask_ds: xr.Dataset | xr.DataArray | None,
) -> xr.Dataset:
    projected_mask = project_mask_to_reference_grid(mask_ds, dataset)
    if projected_mask is None:
        return dataset.earthml.normalize_dims_and_coords()

    aligned_ds, aligned_mask = xr.align(
        dataset.earthml.normalize_dims_and_coords(),
        projected_mask,
        join="inner",
    )
    valid = aligned_mask.to_array().all("variable")
    return aligned_ds.where(valid)
