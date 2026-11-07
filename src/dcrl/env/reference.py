"""The scripted reference agent: runs the generator's recorded fix, then submits.

Used to check the environment: a correct reference must score 1.0 on every training task.
``python -m dcrl.env.reference --split train --n 10000`` runs the check in-process (the code
is the generator's own), which is fast. The sandbox path is checked in the tests on a sample.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from dcrl.env.environment import DataCleaningEnv
from dcrl.env.executor import Executor, TrustedExecutor
from dcrl.splits import SPLITS, generate_split


def reference_episode(task: object, executor: Executor) -> DataCleaningEnv:
    from dcrl.tasks import Task

    assert isinstance(task, Task)
    env = DataCleaningEnv(executor=executor)
    env.reset(task)
    env.run_python(task.fix_program())
    env.submit()
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dcrl.env.reference", description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), default="train")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--executor", choices=["trusted", "sandbox"], default="trusted")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    executor: Executor = TrustedExecutor() if args.executor == "trusted" else _sandbox()
    started = time.perf_counter()
    rewards: list[float] = []
    failures: list[dict[str, object]] = []
    levels: Counter[int] = Counter()
    for task in generate_split(args.split, args.n, args.seed):
        env = reference_episode(task, executor)
        rewards.append(env.reward)
        levels[task.level] += 1
        if env.reward < 1.0 and len(failures) < 50:
            failures.append(
                {
                    "task_id": task.task_id,
                    "table": task.table_type,
                    "level": task.level,
                    "reward": env.reward,
                    "corruptions": [a.name for a in task.corruptions],
                    "hidden": vars(env.scores["hidden"]),
                }
            )
    perfect = sum(r == 1.0 for r in rewards)
    summary = {
        "split": args.split,
        "tasks": len(rewards),
        "perfect": perfect,
        "mean_reward": sum(rewards) / max(1, len(rewards)),
        "seconds": round(time.perf_counter() - started, 1),
        "executor": args.executor,
        "failures": failures,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    for item in failures[:10]:
        print(json.dumps(item, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
    return 0 if perfect == len(rewards) else 1


def _sandbox() -> Executor:
    from dcrl.env.executor import SandboxExecutor

    return SandboxExecutor()


if __name__ == "__main__":
    raise SystemExit(main())
