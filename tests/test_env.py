"""The environment: tools, scoring on the hidden copy, the turn limit and the reference agent."""

from __future__ import annotations

import pandas as pd
import pytest

from dcrl.corruptions import Applied
from dcrl.env import DataCleaningEnv, TrustedExecutor, score_table
from dcrl.env.diagnostics import check_schema, profile
from dcrl.env.reference import reference_episode
from dcrl.splits import make_split_task
from dcrl.tables import TABLE_TYPES
from dcrl.tasks import build_task

TRUSTED = TrustedExecutor()


def new_env(task) -> DataCleaningEnv:  # type: ignore[no-untyped-def]
    env = DataCleaningEnv(executor=TRUSTED)
    env.reset(task)
    return env


@pytest.mark.parametrize("index", range(40))
def test_reference_agent_scores_one(index: int) -> None:
    task = make_split_task("train", 0, index, (1, 2, 3))
    env = reference_episode(task, TRUSTED)
    assert env.reward == 1.0, env.scores["hidden"]
    assert env.scores["own"].reward == 1.0


def test_empty_submission_scores_zero() -> None:
    env = new_env(make_split_task("train", 0, 1, (1,)))
    env.submit()
    assert env.reward == 0.0
    assert env.scores["hidden"].reward == 0.0


def test_submit_ends_the_episode() -> None:
    env = new_env(make_split_task("train", 0, 2, (1,)))
    env.submit()
    assert "ended" in env.run_python("df['x'] = 1")
    assert "ended" in env.submit()


def test_hand_patched_cells_do_not_transfer_to_the_hidden_copy() -> None:
    # Patch every row of one column on the agent's copy with literals from its clean table.
    # Each literal lands on a different row of the hidden copy, so only the agent's copy
    # benefits: its own score is 1.0, the hidden score falls by about the patched share.
    task = build_task("inventory", 1, 17)
    column = "einkaufspreis"
    clean_a = task.a.clean
    lines = [
        f"df.loc[{row}, {column!r}] = {float(clean_a[column].iloc[row])!r}" for row in range(50)
    ]
    env = DataCleaningEnv(executor=TRUSTED)
    env.reset(task)
    env.run_python(task.fix_program())
    env.run_python("\n".join(lines))
    env.submit()
    assert env.scores["own"].reward == 1.0
    assert env.scores["hidden"].reward < 0.9


def test_turn_limit_blocks_tools_but_not_submit() -> None:
    env = new_env(build_task("staff", 1, 4))
    for _ in range(DataCleaningEnv.MAX_TURNS):
        env.run_python("pass")
    assert "Turn limit" in env.run_python("pass")
    assert "Turn limit" in env.profile()
    env.submit()
    assert env.submitted


def test_calls_beyond_four_are_charged() -> None:
    task = build_task("staff", 1, 9)
    env = new_env(task)
    for _ in range(6):
        env.profile("abteilung")
    env.run_python(task.fix_program())
    env.submit()
    # 6 profiles + 1 run + 1 submit = 8 calls, so the penalty is 0.02 * (8 - 4) = 0.08.
    assert env.calls == 8
    assert env.reward == pytest.approx(1.0 - 0.08)


def test_failed_code_leaves_the_table_unchanged() -> None:
    env = new_env(build_task("staff", 1, 5))
    before = env.df.copy(deep=True) if env.df is not None else None
    message = env.run_python("df = df.does_not_exist()")
    assert "df was not changed" in message
    assert before is not None and env.df is not None
    assert env.df.equals(before)
    assert env.history == []


def test_failing_replay_scores_zero() -> None:
    # Code that holds on copy A but not on the hidden copy must score 0. The assertion pins
    # one value of copy A's messy table; copy B has different rows, so it fails there.
    task = build_task("sensor_logs", 1, 6)
    value = task.a.messy()["temperatur"].iloc[0]
    env = new_env(task)
    env.run_python(f"assert df.iloc[0]['temperatur'] == {value!r}")
    env.submit()
    assert env.scores["own"].reward > 0.0
    assert env.reward == 0.0


def test_profile_shows_the_patterns_in_messy_text() -> None:
    task = build_task("invoice_lines", 2, 3, corruptions=[
        Applied("decimal_comma", {"column": "preis_netto"}),
    ])  # fmt: skip
    env = new_env(task)
    report = env.profile("preis_netto")
    assert "comma_decimal" in report
    assert "sample values" in report


def test_check_schema_flags_messy_data_and_passes_the_fix() -> None:
    task = build_task("orders", 2, 11)
    messy_report = check_schema(task.a.messy(), task.schema)
    assert "dtype object" in messy_report
    env = new_env(task)
    env.run_python(task.fix_program())
    assert env.check_schema().startswith("No schema violations")


def test_check_schema_reports_missing_and_extra_columns() -> None:
    task = build_task("payroll", 1, 2)
    df = task.a.messy().drop(columns=["brutto"]).rename(columns={"monat": "extra"})
    report = check_schema(df, task.schema)
    assert "missing columns" in report and "brutto" in report
    assert "extra columns" in report


def test_profile_handles_unknown_column() -> None:
    env = new_env(build_task("staff", 1, 8))
    assert "No column named" in env.profile("nope")


def test_briefing_has_schema_rows_and_no_answer() -> None:
    task = build_task("sensor_logs", 1, 12)
    text = DataCleaningEnv(executor=TRUSTED).reset(task)
    assert "Target schema" in text
    assert "The table has 50 rows" in text
    assert task.fix_program() not in text
    for spec in task.schema.columns:
        assert spec.name in text


def test_score_table_matches_exact_and_partial_results() -> None:
    task = build_task("inventory", 1, 21)
    result = TRUSTED.run(task.fix_program(), task.a.messy()).df
    assert result is not None
    exact = score_table(result, task.a.clean, task.schema, calls=2)
    assert exact.reward == exact.row_f1 * exact.cell_accuracy == 1.0
    dropped = score_table(result.iloc[:-1], task.a.clean, task.schema, calls=2)
    assert dropped.recall == pytest.approx(49 / 50) and dropped.precision == 1.0
    assert dropped.reward < exact.reward
    doubled_frame = pd.concat([result, result.iloc[[0]]], ignore_index=True)
    doubled = score_table(doubled_frame, task.a.clean, task.schema, calls=2)
    assert doubled.precision < 1.0
    assert score_table(None, task.a.clean, task.schema, calls=2).reward == 0.0


def test_every_table_type_has_a_well_formed_profile() -> None:
    for name in TABLE_TYPES:
        task = build_task(name, 1, 30)
        assert profile(task.a.messy())
