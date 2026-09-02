"""Slow-lane async runner.

Dispatches expensive operations as background jobs. When complete, the job
writes its results back as columns via store.write_back_column so follow-up
queries are fast again (they're just column reads at that point).
"""
from __future__ import annotations

import threading
from typing import Any, Callable


def run_async(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    registry: Any,
    store: Any,
    job_id: str,
) -> str:
    """Dispatch a function to run as a background job.

    The job is pre-registered as "running"; a daemon thread runs ``fn`` and, on
    completion, marks the job done (storing the Result) or failed (storing the
    error message). Because analytics functions that produce new data already
    write it back as columns via the store, no extra write-back is needed here —
    the completed Result is what callers poll for.

    Args:
        fn: The analytics function to call asynchronously.
        args: Positional arguments to pass to fn.
        kwargs: Keyword arguments to pass to fn.
        registry: The JobRegistry to update with status and result.
        store: The Store (passed through to fn via args; kept for symmetry).
        job_id: The unique job identifier (pre-registered in registry).

    Returns:
        The job_id, so the caller can poll status via registry.get(job_id).
    """
    def _worker() -> None:
        try:
            result = fn(*args, **kwargs)
            registry.complete(job_id, result)
        except Exception as exc:  # surface the failure on the job, don't crash
            registry.fail(job_id, str(exc))

    threading.Thread(target=_worker, daemon=True).start()
    return job_id
