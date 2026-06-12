from __future__ import annotations

from dataclasses import dataclass

from .adapters import RunResult, load_backend
from .budgets import BudgetGuard
from .config import ExperimentConfig
from .strategies import StrategyPlan, build_strategy_plan


@dataclass
class ExperimentRunner:
    config: ExperimentConfig

    def plan(self) -> StrategyPlan:
        return build_strategy_plan(self.config)

    def run(self, *, dry_run: bool = False) -> RunResult:
        strategy = self.plan()
        BudgetGuard(self.config.budget)
        backend = load_backend(self.config)
        return backend.run(self.config, strategy, dry_run=dry_run)

