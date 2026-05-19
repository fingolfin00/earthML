from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ItemsView, Literal, Sequence

from .constants import (
    CALCULATE_METRICS_ITEMS,
    CONTEXT_AXES,
    METRIC_KIND,
    METRIC_SECTIONS,
    PROFILE_METRIC_MODES,
    SCOREBOARD_MODES,
    VARIABLE_COLORS,
    CalculateMetricsMetricVsDeltaPairConfig,
)
from ...experiments.mlbc.registry import MLBCExperimentMode


def _is_profile_mode_name(value: object) -> bool:
    return isinstance(value, str) and value in PROFILE_METRIC_MODES


def _validate_profile_reduce_spec(reduce_spec: object) -> None:
    if reduce_spec is None:
        return
    if isinstance(reduce_spec, str):
        if not reduce_spec.strip():
            raise ValueError("Profile reduce dims cannot contain empty strings.")
        return
    if isinstance(reduce_spec, (list, tuple)):
        if not reduce_spec:
            raise ValueError("Profile reduce dims must contain at least one dimension or be None.")
        invalid_dims = [dim for dim in reduce_spec if not isinstance(dim, str) or not dim.strip()]
        if invalid_dims:
            raise ValueError(
                "Profile reduce dims must be strings naming existing xarray dimensions."
            )
        return
    raise ValueError("Profile reduce dims must be None, a string, or a sequence of strings.")


def _looks_like_profile_source_spec(metric_mode: object) -> bool:
    if not isinstance(metric_mode, (list, tuple)):
        return False
    if not metric_mode:
        return False
    if all(_is_profile_mode_name(item) for item in metric_mode):
        return False
    if len(metric_mode) == 3:
        metric_type, kind, _ = metric_mode
        return metric_type in PROFILE_METRIC_MODES and kind in METRIC_KIND
    return False


def _validate_profile_metric_mode(metric_name: str, metric_mode: object, *, field_name: str) -> None:
    if isinstance(metric_mode, str):
        if metric_mode not in PROFILE_METRIC_MODES:
            raise ValueError(
                f"Unsupported {field_name}[{metric_name!r}]={metric_mode!r}. "
                f"Expected a legacy profile mode from {PROFILE_METRIC_MODES!r}, "
                "or an explicit (metric_type, kind, reduce) source spec."
            )
        return

    if isinstance(metric_mode, (list, tuple)):
        if not metric_mode:
            raise ValueError(f"{field_name}[{metric_name!r}] must contain at least one profile mode.")

        if _looks_like_profile_source_spec(metric_mode):
            metric_type, kind, reduce_spec = metric_mode
            if metric_type not in PROFILE_METRIC_MODES or kind not in METRIC_KIND:
                raise ValueError(
                    f"Unsupported {field_name}[{metric_name!r}] source spec {(metric_type, kind)!r}. "
                    f"Expected metric_type from {PROFILE_METRIC_MODES!r} and kind from {METRIC_KIND!r}."
                )
            _validate_profile_reduce_spec(reduce_spec)
            return

        for item in metric_mode:
            _validate_profile_metric_mode(metric_name, item, field_name=field_name)
        return

    raise ValueError(
        f"{field_name}[{metric_name!r}] entries must be strings, sequences of strings, "
        "or explicit profile source specs."
    )


@dataclass(slots=True, kw_only=True)
class CalculateMetricsFilters:
    leadtime: list[int] | None
    variable: list[str] | None
    region: list[str] | None
    train_period: list[str] | None
    loss: list[str] | None
    variant: list[str] | None

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self.__dataclass_fields__:
            return default
        return getattr(self, key)

    def __getitem__(self, key: str) -> Any:
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.__dataclass_fields__

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    def items(self) -> ItemsView[str, Any]:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }.items()


@dataclass(slots=True, kw_only=True)
class CalculateMetricsGlobalConfig:
    """
    Top-level runtime settings shared across all metric calculation outputs.
    """

    exp_root_folder: Path
    plot_folder_name: str
    exp_mode: MLBCExperimentMode
    metric_names: list[str] | None
    external_mask_path: Path | None
    external_mask_variable: str | None
    use_saved_mask: bool
    calculate_clim_from_train_period: bool
    use_train_prediction_clim: bool # only used if calculate_clim_from_train_period is True
    items: list[str]
    print_user_config: bool
    print_console_tables: bool

    def __post_init__(self) -> None:
        if not self.plot_folder_name:
            raise ValueError("plot_folder_name must be a non-empty string")

        if self.exp_mode not in ("test", "train"):
            raise ValueError("exp_mode must be 'test' or 'train'")

        if not self.items:
            raise ValueError("items must contain at least one entry")
        invalid_items = [item for item in self.items if item not in CALCULATE_METRICS_ITEMS]
        if invalid_items:
            raise ValueError(
                f"Unsupported items {invalid_items!r}. Expected any of {CALCULATE_METRICS_ITEMS!r}."
            )

    @property
    def plot_root(self) -> Path:
        return self.exp_root_folder / self.plot_folder_name / self.exp_mode


@dataclass(slots=True, kw_only=True)
class CalculateMetricsTimeseriesConfig:
    filters: CalculateMetricsFilters
    combine_leadtimes: bool
    leadtime_style: Literal["shade", "linestyle"]
    plot_realization_members: bool

    def __post_init__(self) -> None:
        if self.leadtime_style not in ("shade", "linestyle"):
            raise ValueError("leadtime_style must be 'shade' or 'linestyle'")


@dataclass(slots=True, kw_only=True)
class CalculateMetricsMapConfig:
    filters: CalculateMetricsFilters
    field_cmap: str
    realization_mode: Literal["mean", "members"]
    metric_cmaps: dict[str, str]
    metric_cmap_default: str
    metric_limits: dict[str, dict[str, dict[str, Any]]]
    lon_tick_step: float
    lat_tick_step: float
    group_leadtimes: bool = False
    average_leadtimes: bool = False
    leadtime_window_size: int = 3
    leadtime_window_stride: int = 1

    def __post_init__(self) -> None:
        if self.realization_mode not in ("mean", "members"):
            raise ValueError("realization_mode must be 'mean' or 'members'")
        if self.group_leadtimes and self.average_leadtimes:
            raise ValueError("group_leadtimes and average_leadtimes are mutually exclusive")
        if self.leadtime_window_size < 1:
            raise ValueError("leadtime_window_size must be >= 1")
        if self.leadtime_window_stride < 1:
            raise ValueError("leadtime_window_stride must be >= 1")


@dataclass(slots=True, kw_only=True)
class CalculateMetricsProfileConfig:
    filters: CalculateMetricsFilters
    x_axis: str | list[str]
    metrics: dict[str, Any]
    shade_by: str
    shade_label: str
    enable_combined_variable_profiles: bool
    combined_variable_metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        x_axis_fields = [self.x_axis] if isinstance(self.x_axis, str) else self.x_axis
        if not x_axis_fields:
            raise ValueError("x_axis must contain at least one axis")
        invalid_x_axis = [field for field in x_axis_fields if field not in CONTEXT_AXES]
        if invalid_x_axis:
            raise ValueError(f"Unsupported x_axis entries {invalid_x_axis!r}. Expected any of {CONTEXT_AXES!r}.")

        if self.shade_by not in CONTEXT_AXES:
            raise ValueError(f"shade_by must be one of {CONTEXT_AXES!r}")

        for metric_name, metric_mode in self.metrics.items():
            _validate_profile_metric_mode(metric_name, metric_mode, field_name="metrics")

        if self.combined_variable_metrics is not None:
            for metric_name, metric_mode in self.combined_variable_metrics.items():
                _validate_profile_metric_mode(
                    metric_name,
                    metric_mode,
                    field_name="combined_variable_metrics",
                )


@dataclass(slots=True, kw_only=True)
class CalculateMetricsVariableColorConfig:
    base_colors: dict[str, str] = field(default_factory=lambda: dict(VARIABLE_COLORS))
    overrides: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class CalculateMetricsSaveConfig:
    output_subfolder: str
    metric_types: tuple[str, ...] = field(default_factory=lambda: tuple(METRIC_SECTIONS))
    kinds: tuple[str, ...] = field(default_factory=lambda: tuple(METRIC_KIND))
    file_format: Literal["netcdf"] = "netcdf"
    include_empty: bool = False
    reuse_existing: bool = False

    def __post_init__(self) -> None:
        if not self.output_subfolder:
            raise ValueError("output_subfolder must be a non-empty string")
        if not self.metric_types:
            raise ValueError("metric_types must contain at least one entry")
        invalid_metric_types = [metric_type for metric_type in self.metric_types if metric_type not in METRIC_SECTIONS]
        if invalid_metric_types:
            raise ValueError(
                f"Unsupported metric_types {invalid_metric_types!r}. Expected any of {METRIC_SECTIONS!r}."
            )
        if not self.kinds:
            raise ValueError("kinds must contain at least one entry")
        invalid_kinds = [kind for kind in self.kinds if kind not in METRIC_KIND]
        if invalid_kinds:
            raise ValueError(
                f"Unsupported kinds {invalid_kinds!r}. Expected any of {METRIC_KIND!r}."
            )
        if self.file_format != "netcdf":
            raise ValueError("file_format must be 'netcdf'")


@dataclass(slots=True, kw_only=True)
class CalculateMetricsTableConfig:
    scalar_filters: CalculateMetricsFilters
    normalized_filters: CalculateMetricsFilters
    row_index: Sequence[str]
    column_index: Sequence[str]
    stat_order: Sequence[str]
    significant_digits: int
    image_dpi: int
    background_color: str


@dataclass(slots=True, kw_only=True)
class CalculateMetricsMetricVsDeltaConfig:
    filters: CalculateMetricsFilters
    metric_pairs: list[CalculateMetricsMetricVsDeltaPairConfig]
    color_by: str | list[str]
    marker_by: str | list[str]
    shade_by: str
    shade_label: str
    point_size: float

    def __post_init__(self) -> None:
        color_by_fields = [self.color_by] if isinstance(self.color_by, str) else self.color_by
        invalid_color_by = [field for field in color_by_fields if field not in CONTEXT_AXES]
        if invalid_color_by:
            raise ValueError(f"Unsupported color_by entries {invalid_color_by!r}. Expected any of {CONTEXT_AXES!r}.")

        marker_by_fields = [self.marker_by] if isinstance(self.marker_by, str) else self.marker_by
        invalid_marker_by = [field for field in marker_by_fields if field not in CONTEXT_AXES]
        if invalid_marker_by:
            raise ValueError(f"Unsupported marker_by entries {invalid_marker_by!r}. Expected any of {CONTEXT_AXES!r}.")

        if self.shade_by not in CONTEXT_AXES:
            raise ValueError(f"shade_by must be one of {CONTEXT_AXES!r}")

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self.__dataclass_fields__:
            return default
        return getattr(self, key)


@dataclass(slots=True, kw_only=True)
class CalculateMetricsScoreboardConfig:
    filters: CalculateMetricsFilters
    mode: Literal["absolute", "relative"]
    metrics: dict[str, str]
    metric_cmaps: dict[str, str]
    metric_vlims: dict[str, tuple[float, float]]
    row_axis: str
    col_axis: str
    annotate: bool

    def __post_init__(self) -> None:
        if self.mode not in SCOREBOARD_MODES:
            raise ValueError(f"mode must be one of {SCOREBOARD_MODES!r}")

        for metric_name, metric_type in self.metrics.items():
            if metric_type not in METRIC_SECTIONS:
                raise ValueError(
                    f"Unsupported metrics[{metric_name!r}]={metric_type!r}. "
                    f"Expected one of {METRIC_SECTIONS!r}."
                )

        if self.row_axis not in CONTEXT_AXES:
            raise ValueError(f"row_axis must be one of {CONTEXT_AXES!r}")
        if self.col_axis not in CONTEXT_AXES:
            raise ValueError(f"col_axis must be one of {CONTEXT_AXES!r}")


@dataclass(slots=True, kw_only=True)
class CalculateMetricsModelConfig:
    truth_model: str
    reference_model: str
    corrected_model: str | None
    gain_label: str
    display_names: dict[str, str]
    model_colors: dict[str, str]

    @property
    def load_models(self) -> list[str]:
        return [
            self.truth_model,
            self.reference_model,
            self.corrected_model,
        ] if self.corrected_model is not None else [
            self.truth_model,
            self.reference_model,
        ]

    @property
    def comparison_models(self) -> list[str]:
        return [
            self.reference_model,
            self.corrected_model,
        ] if self.corrected_model is not None else [
            self.reference_model,
        ]

    @property
    def difference_model(self) -> str | None:
        return f"{self.corrected_model}-{self.reference_model}" if self.corrected_model is not None else None

    @property
    def table_model_order(self) -> list[str]:
        return [
            self.reference_model,
            self.corrected_model,
            self.gain_label,
        ] if self.corrected_model is not None else [
            self.reference_model,
        ]


@dataclass(slots=True, kw_only=True)
class CalculateMetricsConfig:
    global_config: CalculateMetricsGlobalConfig
    models: CalculateMetricsModelConfig
    variable_colors: CalculateMetricsVariableColorConfig | None
    saved_metrics: CalculateMetricsSaveConfig | None
    timeseries: CalculateMetricsTimeseriesConfig | None
    maps: CalculateMetricsMapConfig | None
    profiles: CalculateMetricsProfileConfig | None
    scalar_tables: CalculateMetricsTableConfig | None
    metric_vs_delta: CalculateMetricsMetricVsDeltaConfig | None
    scoreboards: CalculateMetricsScoreboardConfig | None
