"""The sandbox: worker processes, the time limit, the import guard, and the state round trip.

These run real worker processes, so each test takes a second or two.
"""

from __future__ import annotations

import sys
import time

import pandas as pd
import pytest

from dcrl.env import DataCleaningEnv, SandboxExecutor
from dcrl.env.reference import reference_episode
from dcrl.env.worker import decode_frame, encode_frame
from dcrl.splits import make_split_task
from dcrl.tasks import build_task

SANDBOX = SandboxExecutor()


def small_table() -> pd.DataFrame:
    return pd.DataFrame({"a": ["1", "2"], "b": ["x", None]}, dtype=object)


def test_state_round_trips_through_json() -> None:
    df = pd.DataFrame(
        {
            "id": pd.Series([1, None], dtype="Int64"),
            "value": [1.5, float("nan")],
            "text": ["München", None],
            "day": pd.to_datetime(["2026-11-05", None]),
        }
    )
    back = decode_frame(encode_frame(df))
    assert back.dtypes.astype(str).tolist() == df.dtypes.astype(str).tolist()
    pd.testing.assert_frame_equal(back, df)


def test_changes_persist_between_calls() -> None:
    result = SANDBOX.run("df['c'] = df['a'] + '!'", small_table())
    assert result.ok and result.df is not None
    second = SANDBOX.run("df['d'] = df['c'].str.upper()", result.df)
    assert second.ok and second.df is not None
    assert second.df["d"].tolist() == ["1!".upper(), "2!".upper()]


def test_printed_output_is_returned() -> None:
    result = SANDBOX.run("print('hello', len(df))", small_table())
    assert result.ok
    assert result.stdout.strip() == "hello 2"


def test_error_is_reported_and_no_table_returned() -> None:
    result = SANDBOX.run("raise ValueError('boom')", small_table())
    assert not result.ok and result.df is None
    assert result.error == "ValueError: boom"


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import subprocess",
        "import socket",
        "from urllib import request",
        "import importlib",
    ],
)
def test_forbidden_imports_are_refused(code: str) -> None:
    result = SANDBOX.run(code, small_table())
    assert not result.ok
    assert "not allowed in the sandbox" in (result.error or "")


def test_open_is_not_available() -> None:
    result = SANDBOX.run("open('/etc/hostname')", small_table())
    assert not result.ok
    assert "open" in (result.error or "")


def test_network_call_fails() -> None:
    # The host name never resolves, so this fails with or without network access.
    code = "pd.read_csv('http://sandbox-test.invalid/data.csv')"
    result = SANDBOX.run(code, small_table())
    assert not result.ok


def test_infinite_loop_is_killed_after_ten_seconds() -> None:
    started = time.perf_counter()
    result = SANDBOX.run("while True:\n    pass", small_table())
    elapsed = time.perf_counter() - started
    assert not result.ok
    assert "killed" in (result.error or "")
    assert 9.0 <= elapsed < 20.0


@pytest.mark.skipif(sys.platform == "win32", reason="memory limits need POSIX rlimits")
def test_memory_limit_stops_huge_allocations() -> None:
    result = SANDBOX.run("x = np.ones((50_000, 50_000))", small_table())
    assert not result.ok
    assert "MemoryError" in (result.error or "") or "failed" in (result.error or "")


def test_unsupported_dtype_is_reported_to_the_agent() -> None:
    env = DataCleaningEnv(executor=SANDBOX)
    env.reset(build_task("staff", 1, 14))
    before = env.df.copy(deep=True)  # type: ignore[union-attr]
    message = env.run_python("df['x'] = pd.Categorical(['a'] * len(df))")
    assert "dtype category" in message
    assert env.df is not None and env.df.equals(before)


def test_reference_agent_scores_one_in_the_sandbox() -> None:
    for index in range(6):
        task = make_split_task("train", 0, 1000 + index, (1, 2))
        env = reference_episode(task, SANDBOX)
        assert env.reward == 1.0, (task.task_id, env.scores["hidden"])


def test_hidden_copy_replay_runs_in_order() -> None:
    # Two steps, each depending on the previous one: the replay must keep the order.
    task = build_task("staff", 1, 15)
    env = DataCleaningEnv(executor=SANDBOX)
    env.reset(task)
    env.run_python("df = df.assign(tmp=df.iloc[:, 1].str.strip())")
    env.run_python("df = df.drop(columns=['tmp'])")
    env.submit()
    assert env.history == [
        "df = df.assign(tmp=df.iloc[:, 1].str.strip())",
        "df = df.drop(columns=['tmp'])",
    ]
    assert env.scores["hidden"].calls == 3
