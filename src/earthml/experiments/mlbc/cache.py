from collections.abc import Mapping, Sequence
from typing import Any

from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass

from datetime import datetime
import hashlib
import json

import os
from pathlib import Path
import shutil
import sqlite3
import time
from dateutil.relativedelta import relativedelta

from ...base import DataSelection
from ...logging import get_logger
from ...sources import DataSource


logger = get_logger(__name__)

_CACHE_SCHEMA_VERSION = 1
_LOCK_TIMEOUT_SECONDS = 60 * 60
_LOCK_POLL_SECONDS = 1.0
_READY_MARKERS = ("zarr.json", ".zgroup", ".zmetadata")

_SOURCE_CONFIG_EXCLUDE_FIELDS: dict[str, set[str]] = {
    "earthkit": {"earthkit_cache_dir"},
    "juno-local": {"file_open_workers", "cfgrib_idx_path", "chunk_option"},
    "copernicusmarine": {"username", "password"},
}


@dataclass(frozen=True)
class DatasetCacheEntry:
    cache_key: str
    role: str
    store_path: Path
    manifest_path: Path
    identity: dict[str, Any]


class DatasetCacheManager:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stores_root = self.root / "stores" / "source-role"
        self.manifests_root = self.root / "manifests" / "source-role"
        self.locks_root = self.root / "locks"
        self.stores_root.mkdir(parents=True, exist_ok=True)
        self.manifests_root.mkdir(parents=True, exist_ok=True)
        self.locks_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_cache (
                    cache_key TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    source_summary TEXT NOT NULL,
                    variable_summary TEXT NOT NULL,
                    region TEXT NOT NULL,
                    leadtime_summary TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    freq_summary TEXT NOT NULL,
                    store_path TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    identity_json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if is_dataclass(value):
            return DatasetCacheManager._normalize_value(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, relativedelta):
            # Serialize all explicit relativedelta components into a stable mapping.
            return {
                "__type__": "relativedelta",
                "years": value.years,
                "months": value.months,
                "days": value.days,
                "leapdays": value.leapdays,
                "hours": value.hours,
                "minutes": value.minutes,
                "seconds": value.seconds,
                "microseconds": value.microseconds,
                "year": value.year,
                "month": value.month,
                "day": value.day,
                "weekday": str(value.weekday) if value.weekday is not None else None,
                "hour": value.hour,
                "minute": value.minute,
                "second": value.second,
                "microsecond": value.microsecond,
            }
        if isinstance(value, Mapping):
            return {
                str(k): DatasetCacheManager._normalize_value(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [DatasetCacheManager._normalize_value(v) for v in value]
        return value

    @staticmethod
    def _variable_identity(variable: Any) -> Any:
        if isinstance(variable, Sequence) and not isinstance(variable, (str, bytes)):
            return [DatasetCacheManager._variable_identity(v) for v in variable]
        return DatasetCacheManager._normalize_value(variable)

    @staticmethod
    def _selection_identity(selection: DataSelection) -> dict[str, Any]:
        return {
            "variable": DatasetCacheManager._variable_identity(selection.variable),
            "region": DatasetCacheManager._normalize_value(selection.region),
            "period": DatasetCacheManager._normalize_value(selection.period),
        }

    @classmethod
    def _datasource_identity(cls, datasource: DataSource) -> dict[str, Any]:
        return {
            "source": datasource.source,
            "selection": cls._selection_identity(datasource.data_selection),
        }

    @classmethod
    def _source_config_identity(cls, source_name: str, config: Any) -> Any:
        if config is None:
            return None
        normalized = cls._normalize_value(config)
        if not isinstance(normalized, dict):
            return normalized
        exclude = _SOURCE_CONFIG_EXCLUDE_FIELDS.get(source_name, set())
        return {k: v for k, v in normalized.items() if k not in exclude}

    @classmethod
    def build_identity(
        cls,
        *,
        role: str,
        datasource_list: Sequence[DataSource],
        source_configs: Sequence[Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "role": str(role),
            "datasources": [cls._datasource_identity(ds) for ds in datasource_list],
            "source_configs": [
                cls._source_config_identity(ds.source, cfg)
                for ds, cfg in zip(datasource_list, source_configs, strict=True)
            ],
        }

    @staticmethod
    def _hash_identity(identity: dict[str, Any]) -> str:
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    @staticmethod
    def _is_ready_store(path: Path) -> bool:
        return path.is_dir() and any((path / marker).exists() for marker in _READY_MARKERS)

    @staticmethod
    def _extract_period_summary(datasource_identities: Sequence[Mapping[str, Any]]) -> tuple[str, str, str]:
        periods = [ds["selection"]["period"] for ds in datasource_identities]
        start = min(period["start"] for period in periods)
        end = max(period["end"] for period in periods)
        freqs = sorted({period["freq"] for period in periods})
        return start, end, ",".join(freqs)

    @staticmethod
    def _extract_variable_summary(datasource_identities: Sequence[Mapping[str, Any]]) -> str:
        var_names: list[str] = []
        for ds in datasource_identities:
            variable = ds["selection"]["variable"]
            variables = variable if isinstance(variable, Sequence) and not isinstance(variable, (str, bytes)) else [variable]
            var_names.extend(v["name"] if isinstance(v, Mapping) else str(v) for v in variables)
        return ",".join(sorted(set(var_names)))

    @staticmethod
    def _extract_region_summary(datasource_identities: Sequence[Mapping[str, Any]]) -> str:
        regions = [ds["selection"]["region"]["name"] for ds in datasource_identities]
        return ",".join(sorted(set(regions)))

    @staticmethod
    def _extract_leadtime_summary(source_configs: Sequence[Any]) -> str:
        values: list[str] = []
        for cfg in source_configs:
            if isinstance(cfg, Mapping):
                leadtime = cfg.get("leadtime")
            else:
                leadtime = getattr(cfg, "leadtime", None)
            if leadtime is None:
                values.append("none")
                continue
            if isinstance(leadtime, Mapping):
                name = leadtime.get("name", "leadtime")
                unit = leadtime.get("unit", "")
                value = leadtime.get("value", "")
            else:
                name = getattr(leadtime, "name", "leadtime")
                unit = getattr(leadtime, "unit", "")
                value = getattr(leadtime, "value", str(leadtime))
            values.append(f"{name}:{value}:{unit}")
        return ",".join(values)

    def prepare_entry(
        self,
        *,
        role: str,
        datasource_list: Sequence[DataSource],
        source_configs: Sequence[Any],
    ) -> DatasetCacheEntry:
        identity = self.build_identity(
            role=str(role),
            datasource_list=datasource_list,
            source_configs=source_configs,
        )
        cache_key = self._hash_identity(identity)
        role_dir = self.stores_root / str(role)
        role_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = self.manifests_root / str(role)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        return DatasetCacheEntry(
            cache_key=cache_key,
            role=str(role),
            store_path=role_dir / f"{cache_key}.zarr",
            manifest_path=manifest_dir / f"{cache_key}.json",
            identity=identity,
        )

    def lookup(self, entry: DatasetCacheEntry) -> DatasetCacheEntry | None:
        if self._is_ready_store(entry.store_path) and entry.manifest_path.exists():
            self.touch(entry.cache_key)
            return entry
        self.remove_entry(entry.cache_key, delete_store=False)
        return None

    def register(self, entry: DatasetCacheEntry) -> None:
        now = datetime.utcnow().isoformat()
        manifest = {
            "cache_key": entry.cache_key,
            "schema_version": _CACHE_SCHEMA_VERSION,
            "role": entry.role,
            "store_path": str(entry.store_path),
            "created_at": now,
            "identity": entry.identity,
        }
        entry.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        datasource_identities = entry.identity["datasources"]
        start, end, freq_summary = self._extract_period_summary(datasource_identities)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dataset_cache (
                    cache_key, role, source_summary, variable_summary, region, leadtime_summary,
                    period_start, period_end, freq_summary, store_path, manifest_path,
                    identity_json, schema_version, created_at, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    store_path=excluded.store_path,
                    manifest_path=excluded.manifest_path,
                    identity_json=excluded.identity_json,
                    last_used_at=excluded.last_used_at
                """,
                (
                    entry.cache_key,
                    entry.role,
                    ",".join(sorted({ds["source"] for ds in datasource_identities})),
                    self._extract_variable_summary(datasource_identities),
                    self._extract_region_summary(datasource_identities),
                    self._extract_leadtime_summary(entry.identity["source_configs"]),
                    start,
                    end,
                    freq_summary,
                    str(entry.store_path),
                    str(entry.manifest_path),
                    json.dumps(entry.identity, sort_keys=True),
                    _CACHE_SCHEMA_VERSION,
                    now,
                    now,
                ),
            )

    def touch(self, cache_key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE dataset_cache SET last_used_at=? WHERE cache_key=?",
                (datetime.now(datetime.timezone.utc).isoformat(), cache_key),
            )

    def remove_entry(self, cache_key: str, *, delete_store: bool = True) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT store_path, manifest_path FROM dataset_cache WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
            conn.execute("DELETE FROM dataset_cache WHERE cache_key=?", (cache_key,))
        if row is None:
            return
        manifest_path = Path(row["manifest_path"])
        if manifest_path.exists():
            manifest_path.unlink()
        if delete_store:
            store_path = Path(row["store_path"])
            if store_path.exists():
                shutil.rmtree(store_path)

    @contextmanager
    def build_lock(self, cache_key: str):
        lock_dir = self.locks_root / f"{cache_key}.lock"
        start = time.time()
        while True:
            try:
                os.mkdir(lock_dir)
                break
            except FileExistsError:
                if time.time() - start > _LOCK_TIMEOUT_SECONDS:
                    raise TimeoutError(f"Timed out waiting for dataset cache lock {lock_dir}")
                time.sleep(_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            if lock_dir.exists():
                os.rmdir(lock_dir)
