from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .accounting import CostAccounting
from .config import ExperimentConfig
from .strategies import StrategyPlan


@dataclass(frozen=True)
class RunResult:
    status: str
    run_dir: str
    final_score: float | None
    search_iterations: int
    metric_calls: int
    accounting: CostAccounting
    message: str = ""


class ExperimentBackend(Protocol):
    def run(self, config: ExperimentConfig, strategy: StrategyPlan, *, dry_run: bool = False) -> RunResult:
        """Execute one experiment using the configured backend."""


class DryRunBackend:
    def run(self, config: ExperimentConfig, strategy: StrategyPlan, *, dry_run: bool = False) -> RunResult:
        accounting = CostAccounting()
        return RunResult(
            status="dry_run",
            run_dir=config.experiment.output_dir,
            final_score=None,
            search_iterations=0,
            metric_calls=0,
            accounting=accounting,
            message=(
                f"Resolved mode={strategy.mode.value}, selection_mode={strategy.selection_mode}, "
                f"judge_version={strategy.judge_version}, warmup_rollouts={strategy.warmup_rollouts}."
            ),
        )


def load_backend(config: ExperimentConfig) -> ExperimentBackend:
    backend_type = config.experiment.backend.type
    if backend_type == "dry_run":
        return DryRunBackend()
    if backend_type == "artifact":
        from .artifact_backend import ArtifactBackend

        return ArtifactBackend()
    raise ValueError(f"Unknown backend type: {backend_type}")

