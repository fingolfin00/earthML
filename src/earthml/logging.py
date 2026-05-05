import logging
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, Union

from rich.console import Console
from rich.logging import RichHandler


DEFAULT_LOG_FORMAT = "%(message)s"
DEFAULT_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOGGER_NAME = "earthml"
_SHARED_CONSOLE = Console(stderr=True)
_TEXT_CONSOLE = Console(file=sys.stderr, force_terminal=False, color_system=None, width=120)


def _sanitize_path_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "experiment"


def _next_available_log_path(log_path: Union[str, Path]) -> Path:
    candidate = Path(log_path)
    stem = candidate.stem or "experiment"
    suffix = candidate.suffix
    counter = 0
    while True:
        numbered_candidate = candidate.with_name(f"{stem}_{counter:03d}{suffix}")
        if not numbered_candidate.exists():
            return numbered_candidate
        counter += 1


@dataclass(slots=True)
class LoggingPaths:
    experiment_dir: Path
    log_dir: Path
    log_file: Path


@dataclass(slots=True)
class LoggingConfig:
    level: Union[int, str] = logging.INFO
    file_level: Union[int, str] = logging.DEBUG
    logger_name: str = DEFAULT_LOGGER_NAME
    console: Optional[Console] = None
    use_rich_console: bool = True
    console_markup: bool = True
    show_time: bool = False
    show_level: bool = False
    show_path: bool = False
    capture_warnings: bool = True


def build_experiment_logging_paths(
    experiment_root: Union[str, Path],
    experiment_name: str,
    run_name: Optional[str] = None,
    *,
    log_dir_name: str = "logs",
    filename: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> LoggingPaths:
    experiment_dir = Path(experiment_root) / _sanitize_path_fragment(experiment_name)
    log_dir = experiment_dir / log_dir_name
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_fragment = _sanitize_path_fragment(run_name) if run_name else stamp
    log_file = _next_available_log_path(log_dir / (filename or f"{run_fragment}.log"))
    return LoggingPaths(experiment_dir=experiment_dir, log_dir=log_dir, log_file=log_file)


def _coerce_level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    return logging._nameToLevel.get(level.upper(), logging.INFO)


def _remove_managed_handlers(logger: logging.Logger) -> None:
    handlers_to_remove: list[logging.Handler] = []
    for handler in logger.handlers:
        if getattr(handler, "_earthml_managed", False):
            handlers_to_remove.append(handler)

    for handler in handlers_to_remove:
        logger.removeHandler(handler)
        handler.close()


def configure_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    config = config or LoggingConfig()
    console_level = _coerce_level(config.level)
    file_level = _coerce_level(config.file_level)

    logger = logging.getLogger(config.logger_name)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    _remove_managed_handlers(logger)

    if config.use_rich_console:
        console_handler = RichHandler(
            console=config.console or _SHARED_CONSOLE,
            markup=config.console_markup,
            rich_tracebacks=True,
            show_time=config.show_time,
            show_level=config.show_level,
            show_path=config.show_path,
        )
    else:
        console_handler = logging.StreamHandler()

    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    console_handler._earthml_managed = True
    logger.addHandler(console_handler)

    if config.capture_warnings:
        logging.captureWarnings(True)

    return logger


def add_experiment_file_handler(
    logger: logging.Logger,
    log_file: Union[str, Path],
    *,
    level: Optional[Union[int, str]] = logging.DEBUG,
    mode: str = "a",
) -> Path:
    log_path = _next_available_log_path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setLevel(_coerce_level(logging.DEBUG if level is None else level))
    file_handler.setFormatter(logging.Formatter(DEFAULT_FILE_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    file_handler._earthml_managed = True
    file_handler._earthml_file_path = str(log_path)
    logger.addHandler(file_handler)
    return log_path


def remove_experiment_file_handler(
    logger: logging.Logger,
    log_file: Union[str, Path],
) -> None:
    log_path = str(Path(log_file))
    handlers_to_remove = []

    for handler in logger.handlers:
        if getattr(handler, "_earthml_file_path", None) == log_path:
            handlers_to_remove.append(handler)

    for handler in handlers_to_remove:
        logger.removeHandler(handler)
        handler.close()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or DEFAULT_LOGGER_NAME)


def get_console() -> Console:
    return _SHARED_CONSOLE


def _iter_effective_handlers(logger: logging.Logger) -> Iterator[logging.Handler]:
    seen: set[int] = set()
    current: Optional[logging.Logger] = logger

    while current is not None:
        for handler in current.handlers:
            handler_id = id(handler)
            if handler_id in seen:
                continue
            seen.add(handler_id)
            yield handler

        if not current.propagate:
            break

        current = current.parent


def is_console_enabled_for(
    logger: logging.Logger,
    level: Union[int, str],
) -> bool:
    target_level = _coerce_level(level)

    for handler in _iter_effective_handlers(logger):
        if getattr(handler, "_earthml_file_path", None) is not None:
            continue
        if handler.level <= target_level:
            return True

    return False


def _render_to_text(renderable: Any) -> str:
    with _TEXT_CONSOLE.capture() as capture:
        _TEXT_CONSOLE.print(renderable)
    return capture.get().rstrip()


def log_renderable(
    renderable: Any,
    *,
    logger: Optional[logging.Logger] = None,
    level: Union[int, str] = logging.INFO,
) -> None:
    logger = logger or get_logger()
    log_level = _coerce_level(level)
    rendered_text = _render_to_text(renderable)

    rich_handlers = []
    other_handlers = []
    for handler in _iter_effective_handlers(logger):
        if handler.level > log_level:
            continue
        if isinstance(handler, RichHandler):
            rich_handlers.append(handler)
        else:
            other_handlers.append(handler)

    if rich_handlers:
        for handler in rich_handlers:
            handler.console.print(renderable)
    else:
        _SHARED_CONSOLE.print(renderable)

    if rendered_text:
        record = logger.makeRecord(
            logger.name,
            log_level,
            fn="",
            lno=0,
            msg="\n%s",
            args=(rendered_text,),
            exc_info=None,
        )
        for handler in other_handlers:
            handler.handle(record)


@contextmanager
def experiment_logging(
    experiment_root: Union[str, Path],
    experiment_name: str,
    run_name: Optional[str] = None,
    *,
    config: Optional[LoggingConfig] = None,
    log_dir_name: str = "logs",
    filename: Optional[str] = None,
) -> Iterator[Tuple[logging.Logger, LoggingPaths]]:
    """
    Small sketch for experiment-scoped logging.

    Best integration points in this codebase:
    - call `configure_logging(...)` once in `Runtime.start()`
    - attach the file handler in `MLBCExperimentLauncher.run()` once `run_name` is known

    Example:
    ```python
    logger = configure_logging()
    with experiment_logging(exp_root, experiment_name, run_name) as (logger, paths):
        logger.info("Starting run")
    ```
    """
    logger = configure_logging(config)
    paths = build_experiment_logging_paths(
        experiment_root=experiment_root,
        experiment_name=experiment_name,
        run_name=run_name,
        log_dir_name=log_dir_name,
        filename=filename,
    )

    file_path = add_experiment_file_handler(logger, paths.log_file)
    logger.info("Experiment log file: %s", file_path)

    try:
        yield logger, paths
    finally:
        remove_experiment_file_handler(logger, file_path)
