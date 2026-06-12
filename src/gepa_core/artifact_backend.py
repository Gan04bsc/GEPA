from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .accounting import CostAccounting
from .adapters import RunResult
from .config import ExperimentConfig, ModelConfig
from .strategies import StrategyPlan


class ArtifactBackend:
    def run(self, config: ExperimentConfig, strategy: StrategyPlan, *, dry_run: bool = False) -> RunResult:
        command = build_artifact_command(config, strategy, dry_run=dry_run)
        if dry_run:
            output_dir = _resolve_output_dir(config)
            return RunResult(
                status="dry_run",
                run_dir=str(output_dir),
                final_score=None,
                search_iterations=0,
                metric_calls=0,
                accounting=CostAccounting(),
                message=f"GEPA_EXPERIMENT_DIR={output_dir} " + " ".join(command),
            )
        env = os.environ.copy()
        env["GEPA_EXPERIMENT_DIR"] = str(_resolve_output_dir(config))
        completed = subprocess.run(command, check=False, env=env)
        status = "completed" if completed.returncode == 0 else "failed"
        return RunResult(
            status=status,
            run_dir=config.experiment.output_dir,
            final_score=None,
            search_iterations=0,
            metric_calls=0,
            accounting=CostAccounting(),
            message=f"artifact backend exited with code {completed.returncode}",
        )


def build_artifact_command(config: ExperimentConfig, strategy: StrategyPlan, *, dry_run: bool) -> list[str]:
    backend = config.experiment.backend
    artifact_root = _resolve_path(backend.artifact_root, config) if backend.artifact_root else _repo_root()
    script = artifact_root / "scripts" / (
        "run_hybrid_memory_judge.py" if strategy.needs_two_phase_run else "run_experiments.py"
    )
    _require_backend_fields(config)

    command = [
        sys.executable,
        str(script),
        "--bm_idx",
        str(backend.benchmark_index),
        "--benchmark_name",
        config.experiment.benchmark,
        "--num_threads",
        str(config.run.num_threads),
        "--program_idx",
        str(backend.program_index),
        "--prog_name",
        str(backend.program_name),
        "--opt_idx",
        str(backend.optimizer_index),
        "--optim_name",
        str(backend.optimizer_name),
        "--lm_config",
        json.dumps(_model_payload(config.program.optimizer_lm)),
        "--seed",
        str(config.experiment.seed),
        "--setting_name",
        backend.setting_name or config.experiment.name,
        "--retained_validation_fraction",
        str(config.validation.retained_fraction * 100.0),
        "--validation_sampling_mode",
        config.validation.sampling_mode,
    ]
    if dry_run:
        command.append("--dry_run")

    if config.run.cache_policy != "disabled":
        raise ValueError("artifact backend currently supports cache_policy='disabled' only.")
    command.extend(["--resume_incomplete" if config.run.resume else "--no-resume_incomplete"])

    if strategy.needs_two_phase_run:
        command.extend(
            [
                "--memory_protocol_version",
                f"mem_llm_{config.judge.version}",
                "--warmup_search_iterations",
                str(strategy.warmup_rollouts),
            ]
        )
        _append_total_budget(command, config)
    else:
        command.extend(["--selection_mode", strategy.selection_mode])
        _append_single_phase_budget(command, config)

    if config.judge.enabled:
        command.extend(["--judge_lm_config", json.dumps(_model_payload(config.judge.model or config.program.optimizer_lm))])
        command.extend(["--judge_memory_top_k", str(config.judge.memory_top_k)])
        command.extend(["--judge_memory_same_predictor_only" if config.judge.same_predictor_only else "--no-judge_memory_same_predictor_only"])
        if config.judge.strict_learned_guide:
            command.append("--judge_strict_learned_guide")
        if strategy.uses_combined_score:
            combined = config.judge.combined_strategy
            command.extend(
                [
                    "--combined_score_mode",
                    "normalized" if combined.normalize_validation_delta else "direct",
                    "--combined_validation_weight",
                    str(combined.validation_weight),
                    "--combined_judge_weight",
                    str(combined.judge_weight),
                ]
            )

    if not config.run.final_evaluation:
        command.append("--skip_final_evaluation")
    return command


def _resolve_path(raw_path: str | None, config: ExperimentConfig) -> Path:
    if not raw_path:
        return Path()
    path = Path(raw_path).expanduser()
    if path.is_absolute() or config.source_path is None:
        return path
    return (config.source_path.parent / path).resolve()


def _resolve_output_dir(config: ExperimentConfig) -> Path:
    path = Path(config.paths.result_dir).expanduser()
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _append_single_phase_budget(command: list[str], config: ExperimentConfig) -> None:
    if config.budget.max_llm_calls is not None:
        command.extend(["--override_max_total_api_calls", str(config.budget.max_llm_calls)])
    if config.budget.max_search_iterations is not None:
        command.extend(["--override_max_search_iterations", str(config.budget.max_search_iterations)])
    if config.budget.max_metric_calls is not None:
        command.extend(["--override_max_metric_calls", str(config.budget.max_metric_calls)])


def _append_total_budget(command: list[str], config: ExperimentConfig) -> None:
    limits = [
        config.budget.max_llm_calls is not None,
        config.budget.max_search_iterations is not None,
        config.budget.max_metric_calls is not None,
    ]
    if sum(limits) != 1:
        raise ValueError("Two-phase artifact runs require exactly one of max_llm_calls, max_search_iterations, or max_metric_calls.")
    if config.budget.max_llm_calls is not None:
        command.extend(["--total_api_calls", str(config.budget.max_llm_calls)])
    elif config.budget.max_search_iterations is not None:
        command.extend(["--total_search_iterations", str(config.budget.max_search_iterations)])
    elif config.budget.max_metric_calls is not None:
        command.extend(["--total_metric_calls", str(config.budget.max_metric_calls)])


def _model_payload(model: ModelConfig) -> dict[str, str | float | None]:
    payload: dict[str, str | float | None] = {"name": model.name}
    if model.api_base:
        payload["api_base"] = model.api_base
    if model.api_key_env:
        payload["api_key"] = f"env:{model.api_key_env}"
    payload["temperature"] = model.temperature
    return payload


def _require_backend_fields(config: ExperimentConfig) -> None:
    backend = config.experiment.backend
    missing = [
        name
        for name in ("benchmark_index", "program_index", "program_name", "optimizer_index", "optimizer_name")
        if getattr(backend, name) is None
    ]
    if missing:
        raise ValueError(f"backend.type='artifact' requires: {', '.join(missing)}")
