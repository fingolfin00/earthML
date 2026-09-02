import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


DEFAULT_LOGGER_NAME = "earthml"
DEFAULT_LOG_FORMAT = "%(message)s"
DEFAULT_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_SHARED_CONSOLE = Console(stderr=True)

_TEXT_CONSOLE = Console(
    file=sys.stderr,
    force_terminal=False,
    color_system=None,
    width=120,
)


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level

    return logging._nameToLevel.get(
        level.upper(),
        logging.INFO,
    )


def _unwrap_logger(
    logger: "EarthMLLogger | logging.Logger",
) -> logging.Logger:
    if isinstance(logger, EarthMLLogger):
        return logger.raw

    return logger


def _iter_effective_handlers(
    logger: logging.Logger,
):
    seen: set[int] = set()
    current: logging.Logger | None = logger

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


def _remove_managed_handlers(
    logger: logging.Logger,
) -> None:
    handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_earthml_managed", False)
    ]

    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def _render_to_text(
    *args: Any,
    sep: str = " ",
    end: str = "\n",
) -> str:
    with _TEXT_CONSOLE.capture() as capture:
        _TEXT_CONSOLE.print(
            *args,
            sep=sep,
            end=end,
        )

    return capture.get().rstrip("\n")


def log_renderable(
    renderable: Any,
    *,
    logger: "EarthMLLogger | logging.Logger | None" = None,
    level: int | str = logging.INFO,
) -> None:
    logger = logger or get_logger()
    logger.print(
        renderable,
        level=level,
    )


class EarthMLLogger:
    """
    Small wrapper around logging.Logger with print-like output.

    Examples
    --------
    logger.print("Month", month)
    logger.print("loss:", loss)
    logger.print(rich_table)

    logger.info("Starting experiment %s", experiment_name)
    logger.warning("Missing file: %s", path)
    logger.exception("Training failed")
    """

    def __init__(
        self,
        logger: logging.Logger,
    ) -> None:
        self._logger = logger

    @property
    def raw(self) -> logging.Logger:
        return self._logger

    @property
    def name(self) -> str:
        return self._logger.name

    def print(
        self,
        *args: Any,
        sep: str = " ",
        end: str = "\n",
        level: int | str = logging.INFO,
    ) -> None:
        """
        Print values like built-in print(), while also saving plain text
        to file handlers.
        """
        log_level = _coerce_level(level)

        if not self._logger.isEnabledFor(log_level):
            return

        rendered_text = _render_to_text(
            *args,
            sep=sep,
            end=end,
        )

        rich_handlers: list[RichHandler] = []
        other_handlers: list[logging.Handler] = []

        for handler in _iter_effective_handlers(self._logger):
            if handler.level > log_level:
                continue

            if isinstance(handler, RichHandler):
                rich_handlers.append(handler)
            else:
                other_handlers.append(handler)

        if rich_handlers:
            for handler in rich_handlers:
                handler.console.print(
                    *args,
                    sep=sep,
                    end=end,
                )
        else:
            _SHARED_CONSOLE.print(
                *args,
                sep=sep,
                end=end,
            )

        if not rendered_text:
            return

        record = self._logger.makeRecord(
            name=self._logger.name,
            level=log_level,
            fn="",
            lno=0,
            msg="%s",
            args=(rendered_text,),
            exc_info=None,
        )

        for handler in other_handlers:
            handler.handle(record)

    def debug(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._logger.debug(
            message,
            *args,
            **kwargs,
        )

    def info(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._logger.info(
            message,
            *args,
            **kwargs,
        )

    def warning(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._logger.warning(
            message,
            *args,
            **kwargs,
        )

    def error(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._logger.error(
            message,
            *args,
            **kwargs,
        )

    def critical(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._logger.critical(
            message,
            *args,
            **kwargs,
        )

    def exception(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._logger.exception(
            message,
            *args,
            **kwargs,
        )

    def log(
        self,
        level: int | str,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._logger.log(
            _coerce_level(level),
            message,
            *args,
            **kwargs,
        )

    def is_enabled_for(
        self,
        level: int | str,
    ) -> bool:
        return self._logger.isEnabledFor(
            _coerce_level(level)
        )


def configure_logging(
    *,
    level: int | str = logging.INFO,
    logger_name: str = DEFAULT_LOGGER_NAME,
    console: Console | None = None,
    show_time: bool = False,
    show_level: bool = False,
    show_path: bool = False,
    console_markup: bool = True,
    capture_warnings: bool = True,
) -> EarthMLLogger:
    """
    Configure the shared EarthML console logger.

    Call this once near the start of the application.
    """
    console_level = _coerce_level(level)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    _remove_managed_handlers(logger)

    console_handler = RichHandler(
        console=console or _SHARED_CONSOLE,
        markup=console_markup,
        rich_tracebacks=True,
        show_time=show_time,
        show_level=show_level,
        show_path=show_path,
    )

    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(
            DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )
    )

    console_handler._earthml_managed = True
    logger.addHandler(console_handler)

    if capture_warnings:
        logging.captureWarnings(True)

    return EarthMLLogger(logger)


def get_logger(
    name: str | None = None,
) -> EarthMLLogger:
    """
    Return an EarthMLLogger.

    configure_logging() should normally be called once before this.
    """
    return EarthMLLogger(
        logging.getLogger(name or DEFAULT_LOGGER_NAME)
    )


def add_file_handler(
    logger: EarthMLLogger | logging.Logger,
    log_file: str | Path,
    *,
    level: int | str = logging.DEBUG,
    mode: str = "a",
) -> logging.FileHandler:
    """
    Add a file handler to an experiment logger.
    """
    raw_logger = _unwrap_logger(logger)

    log_path = Path(log_file)
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_path = str(log_path.resolve())

    for existing_handler in raw_logger.handlers:
        existing_path = getattr(
            existing_handler,
            "_earthml_file_path",
            None,
        )

        if existing_path == resolved_path:
            if not isinstance(
                existing_handler,
                logging.FileHandler,
            ):
                raise TypeError(
                    f"Existing handler for {log_path} "
                    "is not a FileHandler"
                )

            return existing_handler

    file_handler = logging.FileHandler(
        log_path,
        mode=mode,
        encoding="utf-8",
    )

    file_handler.setLevel(
        _coerce_level(level)
    )

    file_handler.setFormatter(
        logging.Formatter(
            DEFAULT_FILE_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )
    )

    file_handler._earthml_managed = True
    file_handler._earthml_file_path = resolved_path

    raw_logger.addHandler(file_handler)

    return file_handler


def remove_file_handler(
    logger: EarthMLLogger | logging.Logger,
    handler: logging.FileHandler,
) -> None:
    """
    Remove and close a file handler.
    """
    raw_logger = _unwrap_logger(logger)

    if handler in raw_logger.handlers:
        raw_logger.removeHandler(handler)

    handler.close()
