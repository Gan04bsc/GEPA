from __future__ import annotations

import time
from dataclasses import dataclass

from .config import BudgetConfig


@dataclass
class Progress:
    llm_calls: int = 0
    search_iterations: int = 0
    metric_calls: int = 0
    elapsed_seconds: float = 0.0


class BudgetGuard:
    """Centralized stopping logic for API-call, iteration, metric-call, and time budgets."""

    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self.started_at = time.monotonic()

    def snapshot(
        self,
        *,
        llm_calls: int = 0,
        search_iterations: int = 0,
        metric_calls: int = 0,
    ) -> Progress:
        return Progress(
            llm_calls=llm_calls,
            search_iterations=search_iterations,
            metric_calls=metric_calls,
            elapsed_seconds=time.monotonic() - self.started_at,
        )

    def should_stop(self, progress: Progress) -> bool:
        return self.stop_reason(progress) is not None

    def stop_reason(self, progress: Progress) -> str | None:
        checks: tuple[tuple[str, float | int, float | int | None], ...] = (
            ("max_llm_calls", progress.llm_calls, self.config.max_llm_calls),
            ("max_search_iterations", progress.search_iterations, self.config.max_search_iterations),
            ("max_metric_calls", progress.metric_calls, self.config.max_metric_calls),
            ("max_seconds", progress.elapsed_seconds, self.config.max_seconds),
        )
        for name, current, limit in checks:
            if limit is not None and current >= limit:
                return name
        return None

