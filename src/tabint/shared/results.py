"""Structured return contract for all analytics functions.

Note: TrainedModel (in supervised.py) is a *separate* callable type, not a Result.
It bundles fitted preprocessing and exposes .predict() — it is not an inert data
container like Result.
"""
from typing import Any

from pydantic import BaseModel, Field

from .honesty import Trust


class Result(BaseModel):
    """Base structured return for every analytics function.

    Two audiences read this: a human at a REPL (via __repr__) and, later, an
    orchestrating agent (via addressable fields). Keep fields explicit.
    """
    method: str                                          # what was actually run/chosen
    summary: str = ""                                    # one-line plain-language summary
    values: dict[str, Any] = Field(default_factory=dict)   # statistics, scores, etc.
    metadata: dict[str, Any] = Field(default_factory=dict) # assumptions, params used
    trust: Trust | None = None                           # honesty seam: confidence + decline
    artifact: Any = None                                 # optional raw object (model, fitted transform)

    model_config = {"arbitrary_types_allowed": True}

    def __repr__(self) -> str:        # readable for humans
        return f"<Result method={self.method!r} summary={self.summary!r}>"
