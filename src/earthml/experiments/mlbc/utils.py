from datetime import datetime, timedelta
from datetime import time as datetime_time
from dateutil.relativedelta import relativedelta
from typing import Sequence

from ...base.dataclasses import TimeRange


def _var_names(vars: str | Sequence[str]) -> str:
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
