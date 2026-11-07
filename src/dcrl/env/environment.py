"""The environment: one messy table per episode, four tools, a reward on the hidden copy.

The public methods that take part in the episode (profile, run_python, check_schema, submit)
are the tools a model calls. TRL exposes each one as a tool; their docstrings describe them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dcrl.env import diagnostics
from dcrl.env.executor import Executor, SandboxExecutor
from dcrl.env.scoring import Score, score_table
from dcrl.tasks import Task, build_from_record

OUTPUT_CHARS = 2000
HEAD_ROWS = 5


def _clip(text: str) -> str:
    if len(text) <= OUTPUT_CHARS:
        return text
    return text[:OUTPUT_CHARS] + f"\n... ({len(text) - OUTPUT_CHARS} more characters)"


class DataCleaningEnv:
    MAX_TURNS = 10

    def __init__(self, executor: Executor | None = None) -> None:
        self.executor: Executor = executor if executor is not None else SandboxExecutor()
        self._clear()

    def _clear(self) -> None:
        self.task: Task | None = None
        self.df: pd.DataFrame | None = None
        self.history: list[str] = []
        self.calls = 0
        self.submitted = False
        self.reward = 0.0
        self.scores: dict[str, Score] = {}

    def reset(self, source: Task | dict[str, Any]) -> str:
        """Start an episode on one task (a Task, or a manifest record). Returns the briefing."""
        self._clear()
        self.task = source if isinstance(source, Task) else build_from_record(source)
        self.df = self.task.a.messy()
        assert self.df is not None
        return (
            "Clean this table into the target schema. Use profile, run_python, check_schema, "
            "then submit. You have at most "
            f"{self.MAX_TURNS} tool calls before submit.\n\n"
            "Target schema:\n"
            f"{self.task.schema.describe()}\n\n"
            f"The table has {len(self.df)} rows. First 20 rows:\n"
            f"{self.df.head(20).to_csv(index=False)}"
        )

    def _begin(self) -> str | None:
        """Counts one tool call. Returns a refusal message instead of running the tool."""
        if self.task is None:
            raise RuntimeError("call reset() before using the tools")
        if self.submitted:
            return "The episode has ended. No more tool calls are accepted."
        self.calls += 1
        if self.calls > self.MAX_TURNS:
            return f"Turn limit of {self.MAX_TURNS} reached. Call submit() to finish."
        return None

    def profile(self, column: str | None = None) -> str:
        """Profile the table as it is now.

        Args:
            column: a column name, or leave empty to profile every column.

        Returns:
            For each column: its type, the number of empty and distinct values, sample values
            (for one column) and counts of odd patterns such as comma decimals, date formats,
            placeholders, units in cells, extra whitespace and broken umlauts.
        """
        refusal = self._begin()
        if refusal is not None:
            return refusal
        assert self.df is not None
        return _clip(diagnostics.profile(self.df, column))

    def run_python(self, code: str) -> str:
        """Run pandas code that works on the variable df. df keeps the changes.

        Args:
            code: Python code that reads and changes df, a pandas DataFrame. pd (pandas) and
                np (numpy) are already imported. Allowed imports: pandas, numpy, re, datetime
                and a few standard-library modules. No files and no network.

        Returns:
            What the code printed, any error (in which case df is unchanged), the new shape
            and the first rows.
        """
        refusal = self._begin()
        if refusal is not None:
            return refusal
        assert self.df is not None
        result = self.executor.run(code, self.df)
        output = result.stdout.rstrip()
        if not result.ok:
            message = f"Error: {result.error}\n(df was not changed)"
            return _clip(f"{output}\n{message}".strip())
        assert result.df is not None
        self.df = result.df
        self.history.append(code)
        head = self.df.head(HEAD_ROWS).to_csv(index=False)
        summary = f"new shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns\n{head}"
        return _clip(f"{output}\n\n{summary}".strip())

    def check_schema(self) -> str:
        """Check df against the target schema: columns, types, units, allowed values, empties.

        Returns:
            The columns that break the schema, with the number of rows and a few example
            values. It does not say how to fix them.
        """
        refusal = self._begin()
        if refusal is not None:
            return refusal
        assert self.df is not None and self.task is not None
        return _clip(diagnostics.check_schema(self.df, self.task.schema))

    def submit(self) -> str:
        """Finish the episode and hand in df. No tool works after this.

        Returns:
            A confirmation only. The score is computed afterwards on a hidden copy of the table.
        """
        if self.task is None:
            raise RuntimeError("call reset() before using the tools")
        if self.submitted:
            return "The episode has already ended."
        self.calls += 1
        self.submitted = True
        self._score()
        return "Submitted. The episode has ended."

    def _score(self) -> None:
        """Reward: the code that ran on copy A is replayed, in order, on the hidden copy B."""
        assert self.task is not None and self.df is not None
        schema = self.task.schema
        ran = bool(self.history)
        hidden = self.task.b.messy()
        for code in self.history:
            step = self.executor.run(code, hidden)
            if not step.ok or step.df is None:
                hidden_df = None
                break
            hidden = step.df
        else:
            hidden_df = hidden
        self.scores = {
            "hidden": score_table(hidden_df, self.task.b.clean, schema, self.calls, ran_code=ran),
            "own": score_table(self.df, self.task.a.clean, schema, self.calls, ran_code=ran),
        }
        self.reward = self.scores["hidden"].reward
