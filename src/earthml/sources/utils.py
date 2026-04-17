from typing import Callable, TypeVar
from pathlib import Path

from datetime import datetime, timedelta

import re, time, shutil

import earthkit.data as ekd

from ..logging import get_logger


logger = get_logger(__name__)

T = TypeVar("T")


def retry_fetch_after_hdf_err_eks_source(
    fetch_fn: Callable,
    *,
    tries: int = 5,
    base_sleep: float = 1.5,
):
    last_e: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            return fetch_fn()
        except Exception as e:
            last_e = e
            logger.warning("   Attempt %s/%s failed: %s", attempt, tries, e)
            time.sleep(base_sleep * (2 ** (attempt - 1)))
    raise RuntimeError(f"Couldn't fetch requested Earthkit source after {tries} attempts") from last_e


def _extract_nc_path_from_oserror(e: Exception) -> Path | None:
    # netCDF4 often formats it like: "...: '/path/file.nc'"
    m = re.search(r"(/[^'\"]+\.(?:nc|nc4))", str(e))
    return Path(m.group(1)) if m else None

def _set_ekd_cache_dir(cache_dir: str | Path = Path("/tmp/earthkit-cache/")):
    ekd.config.set("cache-policy", "user")
    ekd.config.set("user-cache-directory", cache_dir)

def _get_ekd_cache_dir() -> str:
    return ekd.config.get("user-cache-directory")


def _is_earthkit_empty_source(obj) -> bool:
    try:
        from earthkit.data.sources.empty import EmptySource
    except Exception:
        return False
    return isinstance(obj, EmptySource)

def retry_fetch_after_hdf_err(
    fetch_fn: Callable[[], T],
    *,
    error_re: str | None = None,
    tries: int = 5,
    base_sleep: float = 1.5,
    delete_bad_file: bool = False,
    delete_bad_parent: bool = False,
    manage_earthkit_cache: bool = False,
) -> T:
    pat = re.compile(error_re, re.I) if error_re else None
    last_e: Exception | None = None
    orig_ekd_cache_dir: str | None = _get_ekd_cache_dir() if manage_earthkit_cache else None

    def rmdir(directory):
        shutil.rmtree(Path(directory), ignore_errors=False)

    for attempt in range(1, tries + 1):
        try:
            data = fetch_fn()

            if manage_earthkit_cache and _is_earthkit_empty_source(data):
                logger.warning("   EmptySource returned, setting tmp cache dir (%s/%s)", attempt, tries)
                _set_ekd_cache_dir()  # default tmp cache
                # continue retry loop
            else:
                if orig_ekd_cache_dir is not None:
                    _set_ekd_cache_dir(orig_ekd_cache_dir)
                return data

        except Exception as e:
            last_e = e
            msg = str(e)
            p = _extract_nc_path_from_oserror(e)

            # Decide if this exception is retryable
            retryable = True if pat is None else bool(pat.search(msg))

            if not retryable:
                if orig_ekd_cache_dir:
                    _set_ekd_cache_dir(orig_ekd_cache_dir)
                raise  # non-matching error -> fail fast

            logger.warning("   Attempt %s/%s", attempt, tries)
            if p:
                logger.warning("   Matched retryable error opening %s, wait %ss", p, base_sleep)

                if delete_bad_file and p.exists():
                    try:
                        p.unlink()
                        logger.warning("   -> deleted corrupt cache file: %s", p)
                    except Exception as del_e:
                        logger.warning("   -> failed to delete %s: %s", p, del_e)

                if delete_bad_parent:
                    cache_subdir = p.parent
                    if cache_subdir.exists():
                        try:
                            rmdir(cache_subdir)
                            logger.warning("   -> deleted corrupt cache parent subdir: %s", cache_subdir)
                        except Exception as del_e:
                            logger.warning("   -> failed to delete %s: %s", cache_subdir, del_e)
            else:
                # Retryable but no path extracted: log message
                logger.warning("   Matched retryable error (no .nc path found): %s", msg)

        time.sleep(base_sleep * (2 ** (attempt - 1)))

    if manage_earthkit_cache:
        _set_ekd_cache_dir(orig_ekd_cache_dir)
    raise RuntimeError(f"Failed after {tries} attempts; last error: {last_e!r}") from last_e


def generate_hours(freq_str, output_type='string') -> list:
    """A list of strings of hours""" # TODO extend description
    value = int(freq_str[:-1])
    if freq_str[-1] != 'h':
        raise ValueError("Only 'h' (hours) frequency supported")
    times = []
    current = datetime.strptime("00:00", "%H:%M")
    while current.hour < 24:
        if output_type == 'int':
            times.append(int(current.strftime("%H")))
        else:
            times.append(current.strftime("%H:%M"))
        current += timedelta(hours=value)
        if current.hour == 0:  # wrapped past midnight
            break
    return times
