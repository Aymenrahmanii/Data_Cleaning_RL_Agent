"""Running code on a table: the sandbox for agent code, in-process for generator code only."""

from __future__ import annotations

import datetime
import io
import json
import os
import re
import secrets
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from dcrl.env.worker import RESPONSE_MARK, decode_frame, encode_frame

WORKER_PATH = Path(__file__).with_name("worker.py")
WALL_SECONDS = 10.0
_KEEP_ENV = ("SYSTEMROOT", "PATH", "TEMP", "TMP", "HOME", "LANG", "LC_ALL")


@dataclass(frozen=True, eq=False)
class ExecResult:
    ok: bool
    stdout: str
    error: str | None
    df: pd.DataFrame | None


class Executor(Protocol):
    def run(self, code: str, df: pd.DataFrame) -> ExecResult: ...


class TrustedExecutor:
    """Runs code in this process. For the generator's own reference fixes only.

    Model-written code must never reach this class: it has no sandbox at all.
    """

    def run(self, code: str, df: pd.DataFrame) -> ExecResult:
        out = io.StringIO()
        namespace: dict[str, Any] = {
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "re": re,
            "datetime": datetime,
        }
        try:
            with redirect_stdout(out):
                exec(compile(code, "<fix>", "exec"), namespace)
            result = namespace["df"]
            if not isinstance(result, pd.DataFrame):
                raise TypeError("df must still be a pandas DataFrame")
        except Exception as exc:
            return ExecResult(False, out.getvalue(), f"{type(exc).__name__}: {exc}", None)
        return ExecResult(True, out.getvalue(), None, result)


class SandboxExecutor:
    """Runs code in a fresh worker process with a wall-clock limit.

    Each call starts a new interpreter (-I, so no user site or PYTHON* variables), reads the
    table as JSON, runs the code and writes the table back as JSON. A run that exceeds the
    limit is killed with its whole process.
    """

    def __init__(self, timeout: float = WALL_SECONDS) -> None:
        self.timeout = timeout

    def run(self, code: str, df: pd.DataFrame) -> ExecResult:
        nonce = secrets.token_hex(8)
        request = json.dumps(
            {"code": code, "state": encode_frame(df), "nonce": nonce}, ensure_ascii=True
        )
        env = {k: v for k, v in os.environ.items() if k in _KEEP_ENV}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(WORKER_PATH)],
                input=request.encode("ascii"),
                capture_output=True,
                timeout=self.timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, "", f"the run was killed after {self.timeout:.0f} s", None)

        stdout = proc.stdout.decode("utf-8", errors="replace")
        marker = f"{RESPONSE_MARK}{nonce}>> "
        line = next((ln for ln in reversed(stdout.splitlines()) if ln.startswith(marker)), None)
        if line is None and proc.returncode < 0:
            return ExecResult(
                False,
                "",
                f"the run was killed by the sandbox (signal {-proc.returncode}), most likely "
                "over the CPU or memory limit",
                None,
            )
        if line is None:
            tail = proc.stderr.decode("utf-8", errors="replace")[-300:]
            return ExecResult(
                False, "", f"the sandbox process failed (exit code {proc.returncode}): {tail}", None
            )
        response = json.loads(line[len(marker) :])
        if not response["ok"]:
            return ExecResult(False, response["stdout"], response["error"], None)
        return ExecResult(True, response["stdout"], None, decode_frame(response["state"]))
