"""The public Stuttgart employment table: clean, licensed, round-trips, and held out."""

from __future__ import annotations

import numpy as np
import pytest

from dcrl.compare import mismatches
from dcrl.corruptions import CELL, STRUCTURAL, Applied
from dcrl.pools import usable
from dcrl.public import CC_BY_4, PUBLIC_TABLES
from dcrl.splits import SPLITS, make_split_task
from dcrl.tasks import Task, build_task, run_trusted

NAME = "stuttgart_sv_employment"
TABLE = PUBLIC_TABLES[NAME]


def assert_fix_restores(task: Task) -> None:
    program = task.fix_program()
    for copy in (task.a, task.b):
        result = run_trusted(program, copy.messy())
        problems = mismatches(result, copy.clean, task.schema)
        assert not problems, f"{task.task_id}: {problems}\n{task.corruptions}"


def test_source_is_clean_and_matches_schema() -> None:
    clean = TABLE.clean
    assert len(clean) == 323
    assert list(clean.columns) == TABLE.schema.names
    assert clean["zeile_id"].is_unique
    assert not clean.drop(columns=["zeile_id"]).duplicated().any()
    for spec in TABLE.schema.columns:
        assert not clean[spec.name].isna().any(), spec.name
        if spec.allowed is not None:
            assert set(clean[spec.name]) <= set(spec.allowed)
        if spec.textual:
            assert all(usable(v) for v in clean[spec.name]), spec.name


def test_source_licence_and_attribution_recorded() -> None:
    record = build_task(NAME, 1, 3).record()
    source = record["source"]
    assert source["kind"] == "public"
    assert source["licence_url"] == CC_BY_4
    assert "CC BY 4.0" in source["attribution"]
    assert source["source_url"].startswith("https://opendata.stuttgart.de/")


def test_row_cap_applies_at_every_level() -> None:
    assert build_task(NAME, 1, 0).n_rows == 50
    assert build_task(NAME, 2, 0).n_rows == 200
    assert build_task(NAME, 3, 0).n_rows == 323  # level 3 asks for 1,000; the table has 323


def test_copies_sample_different_rows_from_the_same_table() -> None:
    task = build_task(NAME, 2, 5)
    assert task.a.clean.shape == task.b.clean.shape == (200, 5)
    assert not task.a.clean.equals(task.b.clean)
    assert set(task.a.clean["zeile_id"]) <= set(TABLE.clean["zeile_id"])


@pytest.mark.parametrize("corruption", sorted(CELL))
def test_each_applicable_cell_corruption_roundtrips(corruption: str) -> None:
    applied = 0
    for spec in TABLE.schema.columns:
        corr = CELL[corruption]
        if not corr.applies_to(spec):
            continue
        for seed in range(3):
            rng = np.random.default_rng(seed)
            params = {"column": spec.name, **corr.sample_params(spec, rng)}
            assert_fix_restores(
                build_task(NAME, 2, seed, corruptions=[Applied(corruption, params)])
            )
            applied += 1
    if applied == 0:
        pytest.skip(f"{corruption} does not apply to this table")


@pytest.mark.parametrize("corruption", sorted(STRUCTURAL))
def test_each_applicable_structural_corruption_roundtrips(corruption: str) -> None:
    params = STRUCTURAL[corruption].sample_params(TABLE.schema, np.random.default_rng(1), set())
    if params is None:
        pytest.skip(f"{corruption} does not apply to this table")
    assert_fix_restores(build_task(NAME, 2, 7, corruptions=[Applied(corruption, params)]))


@pytest.mark.parametrize("level", [1, 2, 3])
def test_sampled_tasks_roundtrip(level: int) -> None:
    for seed in range(6):
        assert_fix_restores(build_task(NAME, level, 100 + seed))


def test_spelling_variants_of_districts_and_sectors_stay_distinct() -> None:
    from dcrl.corruptions.text import normal_form

    for spec in TABLE.schema.columns:
        if spec.allowed is not None:
            forms = [normal_form(v) for v in spec.allowed]
            assert len(set(forms)) == len(forms), spec.name


def test_held_out_only() -> None:
    for i in range(80):
        assert make_split_task("heldout", 0, i, (1, 2)).table_type in SPLITS["heldout"].tables
    assert NAME in SPLITS["heldout"].tables
    assert NAME not in SPLITS["train"].tables
    assert NAME not in SPLITS["dev"].tables
    assert NAME not in SPLITS["unseen"].tables
