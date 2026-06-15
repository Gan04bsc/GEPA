from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .config import ExperimentConfig


class ExperimentMode(str, Enum):
    VALIDATION_DECAY = "validation_decay"
    PURE_LLM_JUDGE = "pure_llm_judge"
    WARMUP_THEN_LLM_JUDGE = "warmup_then_llm_judge"
    COMBINED = "combined"


@dataclass(frozen=True)
class StrategyPlan:
    mode: ExperimentMode
    selection_mode: Literal["validation", "llm_judge", "llm_judge_score_aware", "validation_llm_judge_combined"]
    judge_version: str
    warmup_rollouts: int
    uses_validation: bool
    uses_judge: bool
    uses_combined_score: bool
    strict_learned_guide: bool

    @property
    def needs_two_phase_run(self) -> bool:
        return self.uses_judge and self.warmup_rollouts > 0


def build_strategy_plan(config: ExperimentConfig) -> StrategyPlan:
    judge = config.judge
    if not judge.enabled:
        return StrategyPlan(
            mode=ExperimentMode.VALIDATION_DECAY,
            selection_mode="validation",
            judge_version="none",
            warmup_rollouts=0,
            uses_validation=True,
            uses_judge=False,
            uses_combined_score=False,
            strict_learned_guide=False,
        )

    if judge.combined or judge.version == "combined":
        return StrategyPlan(
            mode=ExperimentMode.COMBINED,
            selection_mode="validation_llm_judge_combined",
            judge_version=judge.version,
            warmup_rollouts=judge.warmup_rollouts,
            uses_validation=True,
            uses_judge=True,
            uses_combined_score=True,
            strict_learned_guide=judge.strict_learned_guide,
        )

    selection_mode = "llm_judge_score_aware" if judge.version == "v1" else "llm_judge"
    strict_versions = {"v3", "v4", "v5_rules_only", "v5_rules_fewshot"}
    return StrategyPlan(
        mode=ExperimentMode.WARMUP_THEN_LLM_JUDGE if judge.warmup_rollouts > 0 else ExperimentMode.PURE_LLM_JUDGE,
        selection_mode=selection_mode,
        judge_version=judge.version,
        warmup_rollouts=judge.warmup_rollouts,
        uses_validation=judge.warmup_rollouts > 0,
        uses_judge=True,
        uses_combined_score=False,
        strict_learned_guide=judge.version in strict_versions or judge.strict_learned_guide,
    )
