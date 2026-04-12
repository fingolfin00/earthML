from typing import Any
from dataclasses import is_dataclass, fields as dc_fields
from pathlib import Path

from rich.table import Table as RichTable
from rich.highlighter import ReprHighlighter


class Table:
    """Helper class to create rich Tables from multinested dicts and configs."""

    def __init__(
        self,
        data: Any,
        title: str = None,
        params_name: str = None,
        twocols: bool = False,
        max_depth: int = 4,
    ) -> RichTable:
        # Accept ExperimentConfig (dataclass) or dict; coerce as needed.
        if is_dataclass(data):
            data = self._to_pretty(data, max_depth=max_depth)
        elif not isinstance(data, dict):
            raise TypeError(f"Table expects dict or dataclass; got {type(data).__name__}")

        if len(data.keys()) == 1:
            assert isinstance(next(iter(data.values())), dict)  # there must be data
            data_name = next(iter(data.keys()))
            data = data[data_name]  # promote first inner dict to actual data
            title = data_name if title is None else title

        has_inner_dicts = self._has_inner_dicts(data)
        rich_params = {
            "title": title,
            "show_header": bool(has_inner_dicts and title and not twocols),
        }
        self.table = RichTable(**rich_params)

        rowheads = self._get_rowheads(data)  # check only first inner level
        highligher = ReprHighlighter()
        params_name = "params" if params_name is None else params_name

        if has_inner_dicts and rowheads and not twocols:
            self.table.add_column(params_name, style="magenta")
            for k in data.keys():
                self.table.add_column(str(k), style="cyan")

            row = {}
            for r in rowheads:
                row[r] = []
                for v in data.values():
                    if isinstance(v, dict):
                        # Align by key instead of by positional index because
                        # sibling dicts can have different lengths/orders.
                        row[r].append(highligher(str(v[r])) if r in v else "")
                    else:
                        row[r].append("")

            for r in rowheads:
                self.table.add_row(str(r), *row[r])
        else:
            self.table.add_column(title or "config", style="magenta")
            self.table.add_column("", style="cyan")
            for k, v in data.items():
                self.table.add_row(str(k), highligher(str(v)))

    @staticmethod
    def _callable_label(fn: Any) -> str:
        try:
            name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
            mod = getattr(fn, "__module__", None)
            return f"{mod}.{name}" if mod else str(name)
        except Exception:
            return repr(fn)

    @classmethod
    def _to_pretty(cls, obj: Any, *, max_depth: int, _depth: int = 0) -> Any:
        """Convert arbitrary objects (incl. ExperimentConfig dataclass) into a dict/list structure."""
        if _depth >= max_depth:
            return repr(obj)

        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, Path):
            return str(obj)

        if is_dataclass(obj):
            out = {}
            for f in dc_fields(obj):
                out[f.name] = cls._to_pretty(getattr(obj, f.name), max_depth=max_depth, _depth=_depth + 1)
            return out

        if isinstance(obj, dict):
            return {str(k): cls._to_pretty(v, max_depth=max_depth, _depth=_depth + 1) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [cls._to_pretty(v, max_depth=max_depth, _depth=_depth + 1) for v in obj]

        if callable(obj):
            return cls._callable_label(obj)

        # pydantic-like configs
        for attr in ("model_dump", "dict"):
            if hasattr(obj, attr) and callable(getattr(obj, attr)):
                try:
                    return cls._to_pretty(getattr(obj, attr)(), max_depth=max_depth, _depth=_depth + 1)
                except Exception:
                    pass

        return repr(obj)

    def _has_inner_dicts (self, d: dict) -> bool:
        for v in d.values():
            if isinstance(v, dict):
                return True
            elif isinstance(v, (list, tuple)):
                if any(isinstance(i, dict) and self._has_inner_dicts(i) for i in v):
                    return True
        return False

    def _get_rowheads (self, d: dict, recursive: bool = False) -> list:
        rowheads = []
        for v in d.values():
            if isinstance(v, dict):
                rowheads.extend(map(str, v.keys()))
                if recursive:
                    rowheads.extend(self._get_rowheads(v, recursive))
        return list(dict.fromkeys(rowheads))
