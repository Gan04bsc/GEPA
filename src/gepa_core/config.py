from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


JudgeVersion = Literal["v1", "v2", "v3", "v4", "v5_rules_only", "v5_rules_fewshot", "combined"]
SamplingMode = Literal["fixed"]
CachePolicy = Literal["disabled", "read_only", "read_write"]


@dataclass(frozen=True)
class BackendConfig:
    type: str = "dry_run"
    artifact_root: str | None = None
    benchmark_index: int | None = None
    program_index: int | None = None
    program_name: str | None = None
    optimizer_index: int | None = None
    optimizer_name: str | None = None
    setting_name: str | None = None


@dataclass(frozen=True)
class ExperimentSection:
    name: str
    seed: int
    benchmark: str
    output_dir: str
    backend: BackendConfig = field(default_factory=BackendConfig)


@dataclass(frozen=True)
class RunConfig:
    num_threads: int = 1
    resume: bool = True
    cache_policy: CachePolicy = "disabled"
    final_evaluation: bool = True


@dataclass(frozen=True)
class BudgetConfig:
    max_llm_calls: int | None = None
    max_search_tokens: int | None = None
    max_search_iterations: int | None = None
    max_metric_calls: int | None = None
    max_seconds: float | None = None

    def active_limits(self) -> dict[str, int | float]:
        return {
            key: value
            for key, value in {
                "max_llm_calls": self.max_llm_calls,
                "max_search_tokens": self.max_search_tokens,
                "max_search_iterations": self.max_search_iterations,
                "max_metric_calls": self.max_metric_calls,
                "max_seconds": self.max_seconds,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class ValidationDecayConfig:
    enabled: bool = True
    metric: str = "validation_score"
    min_delta: float = 0.0
    normalize_scores: bool = False


@dataclass(frozen=True)
class ValidationConfig:
    retained_fraction: float = 1.0
    sampling_mode: SamplingMode = "fixed"
    decay: ValidationDecayConfig = field(default_factory=ValidationDecayConfig)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    api_base: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0


@dataclass(frozen=True)
class CombinedStrategyConfig:
    validation_weight: float = 1.0
    judge_weight: float = 1.0
    normalize_validation_delta: bool = True
    tie_breaker: Literal["validation", "judge", "old_prompt"] = "validation"


@dataclass(frozen=True)
class V3Config:
    warmup_teacher_fraction: float = 0.05
    distilled_pair_count: int = 5
    similarity_threshold: float = 0.86
    use_strict_memory: bool = True


@dataclass(frozen=True)
class V5Config:
    max_rules: int = 50
    include_fewshot: bool | None = None
    teacher_pair_count: int = 3
    alignment_pair_count: int = 2


@dataclass(frozen=True)
class JudgeConfig:
    enabled: bool = False
    version: JudgeVersion = "v1"
    combined: bool = False
    warmup_rollouts: int = 0
    strict_learned_guide: bool = False
    memory_top_k: int = 3
    same_predictor_only: bool = True
    model: ModelConfig | None = None
    combined_strategy: CombinedStrategyConfig = field(default_factory=CombinedStrategyConfig)
    v3: V3Config = field(default_factory=V3Config)
    v5: V5Config = field(default_factory=V5Config)


@dataclass(frozen=True)
class ProgramConfig:
    optimizer_lm: ModelConfig
    task_lm: ModelConfig


@dataclass(frozen=True)
class PathConfig:
    data_dir: str
    result_dir: str
    cache_dir: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSection
    run: RunConfig
    budget: BudgetConfig
    validation: ValidationConfig
    judge: JudgeConfig
    program: ProgramConfig
    paths: PathConfig
    source_path: Path | None = None

    def validate(self) -> None:
        if not 0.0 <= self.validation.retained_fraction <= 1.0:
            raise ValueError("validation.retained_fraction must be between 0.0 and 1.0.")
        if self.judge.warmup_rollouts < 0:
            raise ValueError("judge.warmup_rollouts must be >= 0.")
        if self.judge.enabled and self.judge.model is None:
            raise ValueError("judge.model is required when judge.enabled=true.")
        if self.judge.version == "v3" and self.judge.enabled and self.judge.v3.distilled_pair_count <= 0:
            raise ValueError("judge.v3.distilled_pair_count must be positive for v3.")
        if self.judge.version.startswith("v5") and self.judge.enabled:
            if self.judge.v5.max_rules <= 0:
                raise ValueError("judge.v5.max_rules must be positive for v5.")
            if self.judge.warmup_rollouts <= 0:
                raise ValueError("v5 rules protocols require judge.warmup_rollouts > 0.")
        if not self.budget.active_limits():
            raise ValueError("At least one budget limit must be set.")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = parse_config(payload, source_path=config_path)
    config.validate()
    return config


def parse_config(payload: dict[str, Any], source_path: Path | None = None) -> ExperimentConfig:
    experiment_payload = payload.get("experiment", {})
    judge_payload = payload.get("judge", {})
    program_payload = payload.get("program", {})

    config = ExperimentConfig(
        experiment=ExperimentSection(
            name=_required(experiment_payload, "name", "experiment.name"),
            seed=int(_required(experiment_payload, "seed", "experiment.seed")),
            benchmark=_required(experiment_payload, "benchmark", "experiment.benchmark"),
            output_dir=_required(experiment_payload, "output_dir", "experiment.output_dir"),
            backend=BackendConfig(**experiment_payload.get("backend", {})),
        ),
        run=RunConfig(**payload.get("run", {})),
        budget=BudgetConfig(**payload.get("budget", {})),
        validation=ValidationConfig(
            **{
                **{k: v for k, v in payload.get("validation", {}).items() if k != "decay"},
                "decay": ValidationDecayConfig(**payload.get("validation", {}).get("decay", {})),
            }
        ),
        judge=JudgeConfig(
            **{
                **{k: v for k, v in judge_payload.items() if k not in {"model", "combined_strategy", "v3", "v5"}},
                "model": _optional_model(judge_payload.get("model")),
                "combined_strategy": CombinedStrategyConfig(**judge_payload.get("combined_strategy", {})),
                "v3": V3Config(**judge_payload.get("v3", {})),
                "v5": V5Config(**judge_payload.get("v5", {})),
            }
        ),
        program=ProgramConfig(
            optimizer_lm=_model(_required(program_payload, "optimizer_lm", "program.optimizer_lm")),
            task_lm=_model(_required(program_payload, "task_lm", "program.task_lm")),
        ),
        paths=PathConfig(**payload.get("paths", {})),
        source_path=source_path,
    )
    return config


def _required(payload: dict[str, Any], key: str, name: str) -> Any:
    if key not in payload:
        raise ValueError(f"Missing required config field: {name}")
    return payload[key]


def _model(payload: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**payload)


def _optional_model(payload: dict[str, Any] | None) -> ModelConfig | None:
    return None if payload is None else _model(payload)
