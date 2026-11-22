"""Timestamped operation tracing for MCQ generation."""

from contextlib import contextmanager
from datetime import datetime
from time import perf_counter


def _log(message):
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    print(f"[{timestamp}] [MCQ] {message}", flush=True)


def preview(value, limit=220):
    """Compact one-line preview for tracing model inputs and outputs."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


@contextmanager
def operation(name):
    """Log the start and outcome of a pipeline operation with its duration."""
    started = perf_counter()
    _log(f"START {name}")
    try:
        yield
    except BaseException as exc:
        elapsed = perf_counter() - started
        _log(f"ERROR {name} after {elapsed:.3f}s: {type(exc).__name__}: {exc}")
        raise
    else:
        elapsed = perf_counter() - started
        _log(f"DONE {name} in {elapsed:.3f}s")
