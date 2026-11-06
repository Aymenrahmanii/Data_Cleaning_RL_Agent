"""Tables, determinism, the two copies, levels and splits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dcrl.cli import main
from dcrl.compare import frames_match
from dcrl.corruptions import CELL, REGISTRY, STRUCTURAL
from dcrl.corruptions.text import normal_form
from dcrl.pools import POOL_NAMES, pool, usable
from dcrl.splits import (
    HELDOUT_TABLES,
    SPLITS,
    TRAIN_TABLES,
    UNSEEN_CORRUPTIONS,
    make_split_task,
)
from dcrl.tables import TABLE_TYPES
from dcrl.tasks import LEVELS, build_from_record, build_task


def test_twenty_table_types_split_14_6() -> None:
    assert len(TABLE_TYPES) == 20
    assert len(TRAIN_TABLES) == 14
    assert len(HELDOUT_TABLES) == 6
    assert not set(TRAIN_TABLES) & set(HELDOUT_TABLES)


@pytest.mark.parametrize("name", sorted(TABLE_TYPES))
def test_clean_tables_are_well_formed(name: str) -> None:
    table = TABLE_TYPES[name]
    df = table.generate(1000, np.random.default_rng(0))
    assert list(df.columns) == table.schema.names
    assert df.iloc[:, 0].is_unique and not df.iloc[:, 0].isna().any()
    assert not df.duplicated().any()
    for spec in table.schema.columns:
        col = df[spec.name]
        assert col.isna().any() == spec.nullable, spec.name
        values = col.dropna()
        if spec.allowed is not None:
            assert set(values) <= set(spec.allowed)
            forms = [normal_form(v) for v in spec.allowed]
            assert len(set(forms)) == len(forms), f"{spec.name}: variants would collide"
        if spec.textual:
            assert all(usable(v) for v in values), spec.name
        if spec.numeric and spec.min_value is not None:
            assert values.min() >= spec.min_value
        if spec.dtype == "date":
            assert values.dt.year.between(2000, 2068).all()


def test_pools_are_usable_and_stable() -> None:
    for name in POOL_NAMES:
        values = pool(name)
        assert len(values) >= 500, name
        assert all(usable(v) for v in values), name
    assert pool("person")[:3] == pool("person")[:3]


def test_same_arguments_same_task_byte_for_byte() -> None:
    a = build_task("orders", 3, 42)
    b = build_task("orders", 3, 42)
    assert a.record() == b.record()
    assert a.a.messy_csv == b.a.messy_csv and a.b.messy_csv == b.b.messy_csv


def test_copies_share_corruptions_but_not_rows() -> None:
    task = build_task("inventory", 2, 5)
    assert not frames_match(task.a.clean, task.b.clean, task.schema)
    assert task.a.messy_csv != task.b.messy_csv
    assert len(task.a.clean) == len(task.b.clean) == LEVELS[2].n_rows


def test_record_round_trips_through_json() -> None:
    task = make_split_task("train", 0, 11, (1, 2, 3))
    record = json.loads(json.dumps(task.record()))
    rebuilt = build_from_record(record)
    assert rebuilt.record() == task.record()


@pytest.mark.parametrize("level", [1, 2, 3])
def test_levels(level: int) -> None:
    spec = LEVELS[level]
    for seed in range(40):
        table = sorted(TABLE_TYPES)[seed % 20]
        task = build_task(table, level, seed)
        n_struct = sum(a.name in STRUCTURAL for a in task.corruptions)
        assert spec.total[0] <= len(task.corruptions) <= spec.total[1]
        assert spec.structural[0] <= n_struct <= spec.structural[1]
        cell_columns = [a.params["column"] for a in task.corruptions if a.name in CELL]
        assert len(cell_columns) == len(set(cell_columns))
        assert task.n_rows == spec.n_rows


def test_splits_respect_tables_and_corruptions() -> None:
    for split in ("train", "dev", "heldout"):
        for i in range(60):
            task = make_split_task(split, 0, i, (1, 2, 3))
            assert task.table_type in SPLITS[split].tables
            assert not {a.name for a in task.corruptions} & UNSEEN_CORRUPTIONS
    for i in range(60):
        task = make_split_task("unseen", 0, i, (1, 2, 3))
        assert {a.name for a in task.corruptions} & UNSEEN_CORRUPTIONS
        assert task.table_type in TRAIN_TABLES


def test_splits_never_share_tasks() -> None:
    hashes = {
        split: {make_split_task(split, 0, i, (1,)).record()["sha256"]["messy_a"] for i in range(50)}
        for split in SPLITS
    }
    names = list(hashes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not hashes[a] & hashes[b], (a, b)


def test_every_corruption_is_sampled_in_training() -> None:
    seen = set()
    for i in range(300):
        seen.update(a.name for a in make_split_task("train", 0, i, (1, 2, 3)).corruptions)
    assert seen == set(REGISTRY) - UNSEEN_CORRUPTIONS


def test_messy_table_is_all_text() -> None:
    task = build_task("sensor_logs", 3, 8)
    messy = task.a.messy()
    assert all(messy[c].dtype == object for c in messy.columns)
    assert all(isinstance(v, str) for v in messy.to_numpy().ravel())


def test_cli_generate_is_reproducible(tmp_path: pd.DataFrame) -> None:
    hashes = []
    for run in ("one", "two"):
        out = tmp_path / run
        assert main(["generate", "--split", "train", "--n", "30", "--out", str(out)]) == 0
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        hashes.append(meta["manifest_sha256"])
        lines = (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 30
    assert hashes[0] == hashes[1]
