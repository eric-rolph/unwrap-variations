"""ASCII-safe logging for UnwrapVariations.

Everything the add-on prints goes through this module.  Two reasons:

1. **Encoding.**  Blender's console on Windows is cp1252.  Writing a single
   non-ASCII character (a degree sign, an em dash, a smart quote) raises
   ``UnicodeEncodeError`` *inside the logging handler*, which surfaces as a
   confusing "--- Logging error ---" wall of text and can, in an operator's
   ``execute``, abort the operator.  Every record is therefore transliterated to
   plain ASCII before it reaches the stream.
2. **Noise control.**  Blender add-ons share one root logger with Blender's own
   Python and with every other add-on.  A dedicated, non-propagating logger with
   its own handler keeps our output ours and keeps it off by default at DEBUG.

The public surface is deliberately tiny::

    from ..utils.logging import get_logger
    log = get_logger(__name__)
    log.info("generated %d candidates", n)

Nothing here imports :mod:`bpy`, so the module is usable from plain CPython
tests as well as from inside Blender.
"""

from __future__ import annotations

import logging
import sys
import unicodedata
from typing import Any, Iterable

__all__ = [
    "ROOT_LOGGER_NAME",
    "get_logger",
    "configure",
    "set_debug",
    "is_debug",
    "ascii_safe",
    "shutdown",
]

#: Name of the package-level logger every other logger hangs off.
ROOT_LOGGER_NAME = "unwrap_variations"

#: Prefix stamped on every line so a user can grep Blender's console for us.
_PREFIX = "[UVVAR]"

#: Sentinel attribute marking the handler this module installed, so a reload
#: replaces its own handler instead of stacking a second one on every F8.
_HANDLER_TAG = "_uvvar_handler"

#: Characters cp1252 cannot encode but which show up constantly in Python
#: repr()s and in text pasted from documentation.  Mapped explicitly because the
#: NFKD fallback below turns them into nothing at all, which loses meaning.
_TRANSLITERATE = {
    0x2018: "'",
    0x2019: "'",
    0x201C: '"',
    0x201D: '"',
    0x2013: "-",
    0x2014: "--",
    0x2026: "...",
    0x00B0: " deg",
    0x00D7: "x",
    0x2192: "->",
    0x2190: "<-",
    0x00B5: "u",
    0x03BC: "u",
    0x00A0: " ",
    0x2264: "<=",
    0x2265: ">=",
    0x00B1: "+/-",
    0x2022: "*",
}

_debug_enabled = False


def ascii_safe(value: Any) -> str:
    """Return ``value`` as a string containing only printable ASCII.

    Common typographic characters are transliterated to a readable ASCII
    equivalent; anything still outside ASCII after Unicode decomposition is
    replaced with ``?`` rather than dropped, so that a mangled string is
    visibly mangled instead of silently shorter.

    Args:
        value: Any object; non-strings are rendered with :func:`str`.

    Returns:
        A pure-ASCII string safe to write to a cp1252 console.
    """
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:  # noqa: BLE001 - a broken __str__ must not break logging.
        return "<unprintable %s>" % type(value).__name__

    if text.isascii():
        return text

    text = text.translate(_TRANSLITERATE)
    if text.isascii():
        return text

    # Decompose accented letters ("e" + combining acute) and drop the marks.
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "replace").decode("ascii")


class _AsciiFormatter(logging.Formatter):
    """Formatter that guarantees an ASCII-only result.

    The sanitisation happens *after* ``%``-substitution because the arguments
    are where non-ASCII text usually comes from -- an object named with an
    accented character, a file path, a backend's error message.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Return the fully formatted, ASCII-only log line."""
        try:
            text = super().format(record)
        except Exception as exc:  # noqa: BLE001 - bad %-args must not raise.
            text = "%s LOGGING-ERROR %s: %r" % (_PREFIX, exc, record.msg)
        return ascii_safe(text)


class _AsciiStreamHandler(logging.StreamHandler):
    """Stream handler that never lets an encoding error escape.

    :class:`_AsciiFormatter` already guarantees ASCII, but a handler is also
    reached during interpreter shutdown and when ``sys.stdout`` has been
    replaced by something unusual (Blender's own console redirector, a test
    harness).  Swallowing here keeps a logging failure from becoming an
    operator failure.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Write one record, ignoring any stream or encoding failure."""
        try:
            super().emit(record)
        except Exception:  # noqa: BLE001 - logging must never break the add-on.
            pass


def _root() -> logging.Logger:
    """Return the package logger, creating and configuring it on first use."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if not getattr(logger, "_uvvar_configured", False):
        configure(debug=False)
    return logger


def configure(*, debug: bool = False, stream: Any | None = None) -> logging.Logger:
    """Install (or reinstall) the package logger's handler.

    Safe to call repeatedly: any handler this module previously installed is
    removed first, so an add-on reload does not double every line.

    Args:
        debug: When ``True`` the level is ``DEBUG``, otherwise ``INFO``.
        stream: Optional stream override, mainly for tests.  Defaults to
            ``sys.stdout``, which is where Blender's console reads from.

    Returns:
        The configured package logger.
    """
    global _debug_enabled
    _debug_enabled = bool(debug)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                pass

    handler = _AsciiStreamHandler(stream if stream is not None else sys.stdout)
    setattr(handler, _HANDLER_TAG, True)
    handler.setFormatter(_AsciiFormatter(_PREFIX + " %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # Blender's root logger is shared with every other add-on; propagating
    # would duplicate our lines into whatever else is listening.
    logger.propagate = False
    logger._uvvar_configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the package logger.

    Args:
        name: Usually ``__name__``.  The ``unwrap_variations.`` prefix (or the
            ``bl_ext.<repo>.unwrap_variations.`` prefix an extension install
            gets) is stripped so the printed name stays short and identical in
            both install modes.

    Returns:
        A ready-to-use :class:`logging.Logger`.
    """
    root = _root()
    if not name:
        return root
    short = str(name)
    marker = ROOT_LOGGER_NAME + "."
    if marker in short:
        short = short.split(marker, 1)[1]
    elif short == ROOT_LOGGER_NAME or short.endswith("." + ROOT_LOGGER_NAME):
        return root
    return root.getChild(short)


def set_debug(enabled: bool) -> None:
    """Raise or lower the package logger's level.

    Called from the add-on preferences so a user can turn on verbose output
    without restarting Blender.

    Args:
        enabled: ``True`` for ``DEBUG``, ``False`` for ``INFO``.
    """
    global _debug_enabled
    _debug_enabled = bool(enabled)
    _root().setLevel(logging.DEBUG if enabled else logging.INFO)


def is_debug() -> bool:
    """Return whether verbose logging is currently enabled."""
    return _debug_enabled


def shutdown() -> None:
    """Remove the handler this module installed.

    Called from the add-on's ``unregister`` so that disabling the add-on stops
    it writing to the console, and so a subsequent enable starts from a clean
    handler list.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                pass
    logger._uvvar_configured = False  # type: ignore[attr-defined]


def format_lines(lines: Iterable[str], *, limit: int = 8) -> str:
    """Join ``lines`` into one ASCII string, truncating politely.

    Used for operator reports, where an unbounded list of per-candidate
    failures would produce an unreadable status bar.

    Args:
        lines: The messages to join.
        limit: Maximum number of lines to keep.

    Returns:
        A ``'; '``-joined ASCII string, with a ``(+N more)`` suffix when the
        input was truncated.
    """
    items = [ascii_safe(line) for line in lines if line]
    if not items:
        return ""
    if len(items) <= limit:
        return "; ".join(items)
    return "; ".join(items[:limit]) + " (+%d more)" % (len(items) - limit)
