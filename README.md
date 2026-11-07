# Data-Cleaning RL Agent

An environment in which an agent cleans messy German business tables (decimal commas,
mixed date formats, placeholders, mixed units, broken umlauts, duplicate and totals rows)
by writing pandas code, and a small open model trained in it with reinforcement learning.
Every task is a clean table broken on purpose by recorded steps, so every score is exact.

> **Status: week 2 of 7, the environment and sandbox.** The baselines and training come
> next. The results table will open this README once there are results.

## Quick start

```bash
uv sync
uv run dcrl show --split train --index 3 --fix     # one task, its corruptions and the reference fix
uv run dcrl generate --split train --n 10000 --seed 0 --out data/train
uv run pytest -q
```

`generate` writes `manifest.jsonl` (one line per task: table type, level, seed, corruption
parameters, public schema and SHA-256 hashes of both copies) and `meta.json` (library
versions and the manifest hash). Add `--files` to also write each task's CSVs, schema and
reference fix. Any task can be rebuilt from its manifest line with
`dcrl.tasks.build_from_record`.

## How a task is made

1. **Clean table.** One of 20 German business table types (orders, invoice lines,
   inventory, staff, sensor logs, deliveries, payroll, fleet, ...), generated from a seed.
   Names, companies and streets come from Faker `de_DE` pools built once from fixed seeds.
2. **Corruptions.** Sampled with their task-level parameters (which column, which
   placeholder, which date formats); each column gets at most one cell corruption.
3. **Two copies.** Same table type, schema and corruption parameters, different rows. The
   agent works on copy A; the reward reruns its code on hidden copy B, so hand-patched
   cells do not carry over.
4. **Messy export.** The corrupted table is written as CSV and read back with every cell as
   text, as an export would arrive.
5. **Reference fix.** Every corruption writes pandas code that undoes its *pattern*. The
   tests check that this code turns both messy copies back into the clean tables exactly.

| Group | Corruptions |
|---|---|
| Numbers | `decimal_comma`, `thousands_dot`, `mixed_number_format`, `currency_symbol`, `percent_sign` |
| Dates | `german_date`, `two_digit_year`, `mixed_dates`, `excel_serial`, `timezone_suffix` |
| Missing | `missing_placeholders` (k.A., n/a, -, NULL, #NV, 0 where no real zero exists, ...) |
| Units | `unit_in_cell`, `mixed_units` (g/kg, mm/cm/m, ml/l) |
| Text | `spelling_variants`, `whitespace`, `mojibake` |
| Structure | `split_column`, `merge_columns`, `duplicate_rows`, `totals_row`, `repeated_header` |

| Level | Rows | Corruptions |
|---|---|---|
| 1 | 50 | 1 cell corruption |
| 2 | 200 | 2–3 cell corruptions |
| 3 | 1,000 | 4–6, of which 1–3 structural |

## Splits

| Split | Table types | Corruptions |
|---|---|---|
| `train`, `dev` | 14 training types | all except the two held back |
| `heldout` | the other 6 types, plus the public table below | same as training |
| `unseen` | training types | every task includes `excel_serial` or `merge_columns`, never seen in training |

Splits mix a split code into every task seed, so they never share a task.

## Public tables

| Table | Source | Licence | Rows | Used in |
|---|---|---|---|---|
| `stuttgart_sv_employment` | Unternehmen und sozialversicherungspflichtig Beschäftigte, Stuttgart 2020 ([opendata.stuttgart.de](https://opendata.stuttgart.de/dataset/37edbf88-cec4-44fe-b5cf-e1ecc7ee87e6)) | [CC BY 4.0](http://dcat-ap.de/def/licenses/cc-by/4.0) | 323 | `heldout` |

The CSV ships inside the package (`src/dcrl/data/`), and `NOTICE.md` there gives the
attribution. Every manifest line records its table's source and licence. A public table has
fewer rows than level 3 asks for, so level-3 tasks on it have 323 rows instead of 1,000.

## Environment

One episode is one messy table. The agent sees the target schema, the row count and the first
20 rows, then calls four tools:

| Tool | Takes | Returns |
|---|---|---|
| `profile(column)` | a column name, or nothing for all | type, empty and distinct counts, sample values, counts of odd patterns (comma decimals, date formats, placeholders, units, whitespace, broken umlauts) |
| `run_python(code)` | pandas code that changes `df` | printed output, errors (df unchanged), new shape and first rows |
| `check_schema()` | nothing | columns and rows that break the schema, with examples; never the fix |
| `submit()` | nothing | confirmation; the episode ends |

At most 10 tool calls before `submit()`.

**Reward**, computed once on the hidden copy: `R = F1_rows x A_cells - 0.02 x max(0, calls - 4)`.
The code the agent ran on its own copy is replayed, in order, on the hidden copy:

- `A` is the share of cells equal to the clean table (numbers within 0.001, text exact,
  dates to the day, types right). Rows are matched by the id column.
- `F1` is over rows. Dropping real rows and keeping duplicates both cost.
- `calls` counts every tool call, including `submit`.
- `R = 0` when nothing was submitted, when the episode ran no code, or when the code fails on
  the hidden copy. We floor the reward at 0 (the formula can go negative with many calls).

Hand-patched cells do not transfer: the reward is computed from the replayed code, so a
literal written for copy A lands on a different row of copy B.

### Sandbox: what is enforced, and what is not yet

Each `run_python` call runs in a fresh worker process (`python -I`), which reads the table
as JSON and writes the new table back as JSON. Pickle is never used, because unpickling
output from agent code would run that code in the trusted process.

| Limit | Status |
|---|---|
| 10 s wall-clock limit, process killed on overrun | enforced (tested: an infinite loop is killed) |
| 10 s CPU, 1 GB address space | enforced on Linux with `resource` limits; not available on Windows |
| Import allow-list (pandas, numpy and a few standard modules) | enforced for the agent's own imports; tested |
| No `open`, `exec`, `eval` | removed from the builtins; tested |
| No network | **not enforced yet.** A failing host name fails (tested), but a reachable one would work |
| Read-only filesystem, no files outside the table | **not enforced yet** |

The Python-level guards are defence in depth. They are not a security boundary, because a
determined program can get around them. The boundary the reward depends on is the container
described in the plan: no network, a read-only root, CPU and memory limits. Docker's CLI is
installed on the development machine, but the daemon is not running, so the container is
not built yet. **Do not use the reward for model training until the container is in place.**

The reference agent (`dcrl.env.reference`) runs the generator's recorded fix and submits.
On the 10,000 training tasks it must score 1.0 on every task.

## Target schema

Each task comes with column names, types (`int`, `float`, `str`, `category`, `date`), units
(weights in kg, lengths in m, money in EUR, dates as ISO 8601), allowed category values and
nullability. Comparison: numbers within 0.001, text exact, dates to the day, and a column
of the wrong dtype matches nowhere.

## Layout

```
src/dcrl/
  schema.py        target schema (public part shown to the agent)
  pools.py         fixed Faker de_DE value pools
  tables.py        the 20 clean table types
  corruptions/     one module per group; each corruption samples, applies and writes its fix
  tasks.py         sampling, the two copies, the reference fix program, manifests
  splits.py        train / dev / heldout / unseen
  compare.py       cell-by-cell comparison with the clean table
  cli.py           dcrl generate, dcrl show
  env/             tools, sandbox worker, diagnostics, reward, reference agent
```

## Licence

The code is released under the MIT licence (see `LICENSE`). Generated tasks are derived
from Faker output and the table types in this repo. Public open-data tables, once added,
will each record their own licence in the task manifest.

## Still to do in week 1

- More public tables (Destatis GENESIS under DL-DE Zero 2.0 is the target). One CC BY table
  is in so far.
