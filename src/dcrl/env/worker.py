"""Sandbox worker: runs one piece of agent code on a table and returns the new table.

This file runs as ``python -I <path to worker.py>`` in its own process. It is a script, not a
module of the package, so it imports only the standard library, pandas and numpy. It reads one
JSON request on stdin and writes one JSON response on stdout, after a marker that carries a
per-call nonce the agent cannot know.

The import guard and the removed builtins stop mistakes and casual misuse. They are not a
security boundary: Python-level guards can be bypassed by a determined program. The boundary
the reward relies on is the container (no network, read-only root filesystem, CPU and memory
limits). See the README.
"""

from __future__ import annotations

# Preload what the allowed libraries load lazily, before the import guard goes up.
import _strptime  # noqa: F401
import builtins
import contextlib
import importlib
import io
import json
import locale  # noqa: F401
import math
import sys
import time  # noqa: F401
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

ALLOWED_IMPORTS = frozenset(
    {
        "pandas",
        "numpy",
        "re",
        "datetime",
        "math",
        "decimal",
        "fractions",
        "unicodedata",
        "collections",
        "itertools",
        "functools",
        "operator",
        "string",
        "json",
        "statistics",
        "calendar",
        # datetime.strptime loads these lazily from C. They do no I/O and are preloaded below.
        "_strptime",
        "locale",
        "time",
    }
)
BLOCKED_BUILTINS = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "input",
        "breakpoint",
        "help",
        "exit",
        "quit",
        "globals",
        "locals",
        "vars",
        "memoryview",
    }
)
CPU_SECONDS = 10
MEMORY_BYTES = 1024**3
RESPONSE_MARK = "<<DCRL-RESPONSE:"
STATE_DTYPES = ("object", "string", "Int64", "int64", "float64", "datetime64[ns]")


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def encode_frame(df: pd.DataFrame) -> dict[str, Any]:
    """A table as plain JSON: column name, dtype and values. Nothing executable crosses over."""
    columns = []
    for pos, name in enumerate(df.columns):
        series = df.iloc[:, pos]
        dtype = str(series.dtype)
        if dtype not in STATE_DTYPES:
            raise TypeError(
                f"column {name!r} has dtype {dtype}; convert it to text, whole numbers, "
                "decimals or dates"
            )
        if dtype == "datetime64[ns]":
            values = [None if _missing(v) else pd.Timestamp(v).isoformat() for v in series]
        elif dtype in ("object", "string"):
            values = [None if _missing(v) else str(v) for v in series]
        elif dtype in ("Int64", "int64"):
            values = [None if _missing(v) else int(v) for v in series]
        else:
            values = [None if _missing(v) else float(v) for v in series]
        columns.append({"name": str(name), "dtype": dtype, "values": values})
    return {"columns": columns}


def decode_frame(state: dict[str, Any]) -> pd.DataFrame:
    """The inverse of encode_frame. Only the dtypes listed in STATE_DTYPES are accepted."""
    series: list[pd.Series] = []
    names: list[str] = []
    for col in state["columns"]:
        dtype, values = col["dtype"], col["values"]
        if dtype == "datetime64[ns]":
            s = pd.to_datetime(pd.Series(values, dtype=object)).astype("datetime64[ns]")
        elif dtype == "object":
            s = pd.Series(values, dtype=object)
        elif dtype == "string":
            s = pd.Series(values, dtype="string")
        elif dtype == "Int64":
            s = pd.Series(values, dtype="Int64")
        elif dtype == "int64":
            s = pd.Series(values, dtype="int64")
        elif dtype == "float64":
            s = pd.Series([math.nan if v is None else v for v in values], dtype="float64")
        else:
            raise ValueError(f"unknown dtype in the table state: {dtype}")
        series.append(s)
        names.append(col["name"])
    df = pd.concat(series, axis=1) if series else pd.DataFrame()
    df.columns = pd.Index(names, dtype=object)
    return df


_real_import: Callable[..., Any] = builtins.__import__


def _guarded_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    # Only imports written by the agent are checked. Libraries that import lazily (for example
    # datetime importing _strptime) run under their own module names and are allowed.
    importer = (globals or {}).get("__name__")
    if importer == "agent" and (level or name.partition(".")[0] not in ALLOWED_IMPORTS):
        raise ImportError(f"import of {name!r} is not allowed in the sandbox")
    return _real_import(name, globals, locals, fromlist, level)


SAFE_BUILTINS: dict[str, Any] = {
    k: v for k, v in vars(builtins).items() if k not in BLOCKED_BUILTINS
}
SAFE_BUILTINS["__import__"] = _guarded_import


def _rlimits() -> Any:
    try:
        return importlib.import_module("resource")
    except ImportError:  # Windows has no rlimits; the parent's wall-clock timeout still applies
        return None


def _limit_cpu() -> None:
    resource = _rlimits()
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))


def _limit_memory() -> None:
    resource = _rlimits()
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))


def main() -> int:
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    nonce = request["nonce"]
    _limit_cpu()
    _limit_memory()
    out = io.StringIO()
    try:
        df = decode_frame(request["state"])
        namespace: dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "agent",
            "pd": pd,
            "np": np,
            "df": df,
        }
        with contextlib.redirect_stdout(out):
            exec(compile(request["code"], "<agent>", "exec"), namespace)
        result = namespace.get("df")
        if not isinstance(result, pd.DataFrame):
            raise TypeError("df must still be a pandas DataFrame")
        response: dict[str, Any] = {
            "ok": True,
            "error": None,
            "stdout": out.getvalue(),
            "state": encode_frame(result),
        }
    except BaseException as exc:  # report everything, including MemoryError and SystemExit
        response = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": out.getvalue(),
            "state": None,
        }
    payload = f"\n{RESPONSE_MARK}{nonce}>> " + json.dumps(response, ensure_ascii=True) + "\n"
    sys.stdout.buffer.write(payload.encode("ascii"))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
