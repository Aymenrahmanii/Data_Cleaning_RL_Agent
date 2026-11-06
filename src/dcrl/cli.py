"""Command line: ``dcrl generate`` writes a split's manifest, ``dcrl show`` prints a task."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

from dcrl.splits import SPLITS, generate_split, make_split_task
from dcrl.tasks import Task


def _levels(text: str) -> tuple[int, ...]:
    levels = tuple(int(x) for x in text.split(","))
    if not levels or any(lv not in (1, 2, 3) for lv in levels):
        raise argparse.ArgumentTypeError("levels are a comma-separated subset of 1,2,3")
    return levels


def _write_files(task: Task, out: Path) -> None:
    folder = out / "tasks" / task.task_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "messy_a.csv").write_text(task.a.messy_csv, encoding="utf-8")
    (folder / "messy_b.csv").write_text(task.b.messy_csv, encoding="utf-8")
    (folder / "clean_a.csv").write_text(task.a.clean_csv(), encoding="utf-8")
    (folder / "clean_b.csv").write_text(task.b.clean_csv(), encoding="utf-8")
    (folder / "fix.py").write_text(task.fix_program(), encoding="utf-8")
    (folder / "schema.json").write_text(
        json.dumps(task.schema.public(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cmd_generate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    digest = hashlib.sha256()
    started = time.perf_counter()
    with manifest.open("w", encoding="utf-8", newline="\n") as fh:
        for i, task in enumerate(generate_split(args.split, args.n, args.seed, args.levels), 1):
            line = json.dumps(task.record(), ensure_ascii=False, sort_keys=True) + "\n"
            fh.write(line)
            digest.update(line.encode("utf-8"))
            if args.files:
                _write_files(task, out)
            if i % 500 == 0:
                print(f"{i}/{args.n} tasks, {time.perf_counter() - started:.0f} s", file=sys.stderr)
    meta = {
        "split": args.split,
        "n": args.n,
        "seed": args.seed,
        "levels": list(args.levels),
        "manifest_sha256": digest.hexdigest(),
        "versions": {p: version(p) for p in ("dcrl", "pandas", "numpy", "faker")},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.n} tasks to {manifest} in {time.perf_counter() - started:.0f} s")
    print(f"manifest sha256 {meta['manifest_sha256']}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    task = make_split_task(args.split, args.seed, args.index, args.levels)
    print(f"{task.task_id}: {task.table_type}, level {task.level}, {task.n_rows} rows")
    print("\nTarget schema:\n" + task.schema.describe())
    print("\nCorruptions:")
    for a in task.corruptions:
        print(f"- {a.name} {json.dumps(a.params, ensure_ascii=False)}")
    print("\nMessy copy A, first rows:")
    print("\n".join(task.a.messy_csv.splitlines()[: args.rows + 1]))
    if args.fix:
        print("\nReference fix:\n" + task.fix_program())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dcrl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a split's tasks into a manifest")
    gen.add_argument("--split", choices=sorted(SPLITS), default="train")
    gen.add_argument("--n", type=int, default=10_000)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--levels", type=_levels, default=(1, 2, 3))
    gen.add_argument("--out", default="data/train")
    gen.add_argument("--files", action="store_true", help="also write each task's CSV files")
    gen.set_defaults(func=cmd_generate)

    show = sub.add_parser("show", help="print one task")
    show.add_argument("--split", choices=sorted(SPLITS), default="train")
    show.add_argument("--index", type=int, default=0)
    show.add_argument("--seed", type=int, default=0)
    show.add_argument("--levels", type=_levels, default=(1, 2, 3))
    show.add_argument("--rows", type=int, default=10)
    show.add_argument("--fix", action="store_true", help="also print the reference fix")
    show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
