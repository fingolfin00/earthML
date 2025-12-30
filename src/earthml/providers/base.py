from typing import Any, Mapping

def merge (base: dict, overrides: Mapping[str, Any] | None = None, **kw) -> dict:
    """Shallow merge, ok for source_params dicts."""
    out = dict(base)
    if overrides:
        out.update(overrides)
    out.update(kw)
    return out

def require_keys (d: Mapping[str, Any], keys: list[str], name: str = "provider") -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{name}: missing required keys: {missing}")
