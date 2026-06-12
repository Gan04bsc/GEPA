import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SRC_ROOT = PROJECT_ROOT / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.experiment_configs import BASE_EXPERIMENT_DIR
    from scripts.run_experiments import (
        bootstrap_openai_compatible_env,
        load_json_from_path,
        resolve_api_key_env_vars,
        run_experiment_and_write_results,
        write_json_to_path,
    )
else:
    from .experiment_configs import BASE_EXPERIMENT_DIR
    from .run_experiments import (
        bootstrap_openai_compatible_env,
        load_json_from_path,
        resolve_api_key_env_vars,
        run_experiment_and_write_results,
        write_json_to_path,
    )

from gepa_artifact.gepa.judge_memory import build_learned_judge_guide, load_memory_bank


def run_dir_for(seed: int, benchmark_name: str, prog_name: str, optim_name: str, lm_name: str, setting_name: str) -> Path:
    run_name = f"{benchmark_name}_{prog_name}_{optim_name}_{lm_name}"
    if setting_name:
        run_name = f"{run_name}__{setting_name}"
    return Path(BASE_EXPERIMENT_DIR) / "experiment_runs" / f"seed_{seed}" / run_name


def copy_required_artifacts(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    file_names = [
        "gepa_state.bin",
        "iteration_summary.jsonl",
        "metric_call_checkpoints.jsonl",
        "instruction_proposer_inpouts.jsonl",
        "judge_memory_bank.jsonl",
        "judge_alignment_memory_bank.jsonl",
        "judge_prompt_lessons.md",
        "judge_prompt_lessons.json",
    ]
    for file_name in file_names:
        src_path = src_dir / file_name
        if src_path.exists():
            shutil.copy2(src_path, dst_dir / file_name)

    src_prog_candidates = src_dir / "prog_candidates"
    if src_prog_candidates.exists():
        shutil.copytree(src_prog_candidates, dst_dir / "prog_candidates", dirs_exist_ok=True)


def collapse_validation_frontier_to_surrogate(dst_dir: Path, warmup_setting_name: str) -> dict:
    state_path = dst_dir / "gepa_state.bin"
    with state_path.open("rb") as f:
        state = pickle.load(f)

    raw_tracked_scores = list(state.get("per_program_tracked_scores") or state.get("program_full_scores_val_set") or [])
    if not raw_tracked_scores:
        raise ValueError(f"No tracked scores found in {state_path}")

    best_score = max(raw_tracked_scores)
    best_programs = sorted(idx for idx, score in enumerate(raw_tracked_scores) if score == best_score)
    primary_program = best_programs[0]
    shifted_scores = [round(score - best_score, 12) for score in raw_tracked_scores]
    state["program_full_scores_val_set"] = shifted_scores
    state["per_program_tracked_scores"] = shifted_scores
    state["pareto_front_valset"] = [0.0]
    state["program_at_pareto_front_valset"] = [{primary_program}]
    state["hybrid_handoff"] = {
        "warmup_setting_name": warmup_setting_name,
        "collapsed_to_surrogate_frontier": True,
        "validation_best_score_at_handoff": best_score,
        "best_programs_at_handoff": best_programs,
        "primary_program_at_handoff": primary_program,
        "surrogate_shift_applied": best_score,
    }

    with state_path.open("wb") as f:
        pickle.dump(state, f)

    return {
        "validation_best_score_at_handoff": best_score,
        "best_programs_at_handoff": best_programs,
        "primary_program_at_handoff": primary_program,
    }


def prepare_continuation_dir(
    warmup_run_dir: Path,
    final_run_dir: Path,
    warmup_setting_name: str,
    memory_protocol_version: str,
) -> dict:
    if (final_run_dir / "gepa_state.bin").exists() and (final_run_dir / "prog_candidates").exists():
        return load_json_from_path(final_run_dir / "hybrid_phase_manifest.json", default={})

    if final_run_dir.exists():
        shutil.rmtree(final_run_dir)

    copy_required_artifacts(warmup_run_dir, final_run_dir)
    collapse_info = collapse_validation_frontier_to_surrogate(final_run_dir, warmup_setting_name)
    manifest = {
        "protocol": memory_protocol_version,
        "protocol_family": "hybrid_memory_judge",
        "warmup_run_dir": str(warmup_run_dir),
        "continuation_run_dir": str(final_run_dir),
        "warmup_setting_name": warmup_setting_name,
        **collapse_info,
    }
    write_json_to_path(final_run_dir / "hybrid_phase_manifest.json", manifest)
    return manifest


def write_learned_judge_guide_artifacts(run_dir: Path, max_cases: int) -> dict:
    memory_path = run_dir / "judge_memory_bank.jsonl"
    records = load_memory_bank(str(memory_path))
    guide = build_learned_judge_guide(records, max_cases=max_cases)
    write_json_to_path(run_dir / "judge_prompt_lessons.json", guide)
    with (run_dir / "judge_prompt_lessons.md").open("w", encoding="utf-8") as f:
        f.write(guide["guide_text"] + "\n")
    return {
        "learned_judge_guide_path": str(run_dir / "judge_prompt_lessons.md"),
        "learned_judge_guide_json_path": str(run_dir / "judge_prompt_lessons.json"),
        "learned_judge_selected_case_count": guide["selected_case_count"],
        "learned_judge_warmup_record_count": guide["warmup_record_count"],
    }


def build_combined_summary(
    *,
    final_setting_name: str,
    memory_protocol_version: str,
    warmup_summary: dict,
    final_summary: dict,
    phase_manifest: dict,
) -> dict:
    warmup_metric_calls = warmup_summary.get("actual_metric_calls", 0) or 0
    final_metric_calls = final_summary.get("actual_metric_calls", 0) or 0
    warmup_iterations = warmup_summary.get("actual_search_iterations", 0) or 0
    final_iterations = final_summary.get("actual_search_iterations", 0) or 0
    combined_validation_tokens = max(
        warmup_summary.get("validation_tokens", 0) or 0,
        final_summary.get("validation_tokens", 0) or 0,
    )
    warmup_search_tokens = warmup_summary.get("search_total_tokens", warmup_summary.get("total_tokens", 0)) or 0
    final_search_tokens = final_summary.get("search_total_tokens", final_summary.get("total_tokens", 0)) or 0
    warmup_search_api_calls = warmup_summary.get("search_total_api_calls", 0) or 0
    final_search_api_calls = final_summary.get("search_total_api_calls", 0) or 0

    return {
        "setting": final_setting_name,
        "protocol": memory_protocol_version,
        "protocol_family": "hybrid_memory_judge",
        "warmup_setting_name": phase_manifest.get("warmup_setting_name"),
        "warmup_run_dir": phase_manifest.get("warmup_run_dir"),
        "continuation_run_dir": phase_manifest.get("continuation_run_dir"),
        "warmup_actual_search_iterations": warmup_iterations,
        "warmup_actual_metric_calls": warmup_metric_calls,
        "post_switch_search_iterations": max(0, final_iterations - warmup_iterations),
        "post_switch_metric_calls": max(0, final_metric_calls - warmup_metric_calls),
        "actual_search_iterations": final_iterations,
        "actual_metric_calls": final_metric_calls,
        "accepted_updates": final_summary.get("accepted_updates"),
        "final_test_score": final_summary.get("final_test_score"),
        "warmup_validation_tokens": warmup_summary.get("validation_tokens", 0),
        "combined_validation_tokens": combined_validation_tokens,
        "phase1_total_tokens": warmup_summary.get("total_tokens", 0),
        "phase2_total_tokens": final_summary.get("total_tokens", 0),
        "reportable_total_tokens": (warmup_summary.get("total_tokens", 0) or 0) + (final_summary.get("total_tokens", 0) or 0),
        "phase1_search_total_tokens": warmup_search_tokens,
        "phase2_search_total_tokens": final_search_tokens,
        "reportable_search_total_tokens": warmup_search_tokens + final_search_tokens,
        "phase1_search_total_api_calls": warmup_search_api_calls,
        "phase2_search_total_api_calls": final_search_api_calls,
        "reportable_search_total_api_calls": warmup_search_api_calls + final_search_api_calls,
        "phase1_minibatch_tokens": warmup_summary.get("minibatch_tokens", 0),
        "phase2_minibatch_tokens": final_summary.get("minibatch_tokens", 0),
        "reportable_minibatch_tokens": (warmup_summary.get("minibatch_tokens", 0) or 0) + (final_summary.get("minibatch_tokens", 0) or 0),
        "phase1_optimization_control_tokens": warmup_summary.get("optimization_control_tokens", 0),
        "phase2_optimization_control_tokens": final_summary.get("optimization_control_tokens", 0),
        "reportable_optimization_control_tokens": (warmup_summary.get("optimization_control_tokens", 0) or 0) + (final_summary.get("optimization_control_tokens", 0) or 0),
        "phase1_total_time_seconds": warmup_summary.get("total_time_seconds", 0),
        "phase2_total_time_seconds": final_summary.get("total_time_seconds", 0),
        "reportable_total_time_seconds": (warmup_summary.get("total_time_seconds", 0) or 0) + (final_summary.get("total_time_seconds", 0) or 0),
        "validation_best_score_at_handoff": phase_manifest.get("validation_best_score_at_handoff"),
        "best_programs_at_handoff": phase_manifest.get("best_programs_at_handoff"),
        "primary_program_at_handoff": phase_manifest.get("primary_program_at_handoff"),
        "validation_sidecar_judge_alignment": phase_manifest.get("validation_sidecar_judge_alignment", False),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run the two-phase hybrid memory-judge GEPA protocol.")
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--bm_idx", type=int, required=True)
    parser.add_argument("--benchmark_name", type=str, required=True)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--program_idx", type=int, required=True)
    parser.add_argument("--prog_name", type=str, required=True)
    parser.add_argument("--opt_idx", type=int, required=True)
    parser.add_argument("--optim_name", type=str, required=True)
    parser.add_argument("--lm_config", type=json.loads, required=True)
    parser.add_argument("--use_cache_from_opt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--setting_name", type=str, required=True)
    parser.add_argument("--retained_validation_fraction", type=float, default=100.0)
    parser.add_argument("--judge_lm_config", type=json.loads, default=None)
    parser.add_argument("--judge_memory_top_k", type=int, default=3)
    parser.add_argument("--judge_memory_same_predictor_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--memory_protocol_version",
        type=str,
        default="mem_llm_v1",
        choices=["mem_llm_v1", "mem_llm_v2", "mem_llm_v3"],
    )
    parser.add_argument("--warmup_metric_calls", type=int, default=None)
    parser.add_argument("--warmup_search_iterations", type=int, default=None)
    parser.add_argument("--total_metric_calls", type=int, default=None)
    parser.add_argument("--total_api_calls", type=int, default=None)
    parser.add_argument("--total_search_iterations", type=int, default=None)
    parser.add_argument("--resume_incomplete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation_sampling_mode", type=str, default="random_per_iteration",
                        choices=["fixed", "random_per_iteration"])
    parser.add_argument("--smoke_train_size", type=int, default=None)
    parser.add_argument("--smoke_val_size", type=int, default=None)
    parser.add_argument("--smoke_test_size", type=int, default=None)
    args = parser.parse_args()
    bootstrap_openai_compatible_env()
    args.lm_config = resolve_api_key_env_vars(args.lm_config)
    args.judge_lm_config = resolve_api_key_env_vars(args.judge_lm_config)
    return args


def validate_budget_args(args) -> tuple[dict, dict, str]:
    if args.memory_protocol_version in {"mem_llm_v1", "mem_llm_v2"}:
        if args.warmup_metric_calls is None or args.total_metric_calls is None:
            raise ValueError("mem_llm_v1/v2 require --warmup_metric_calls and --total_metric_calls.")
        if args.warmup_metric_calls >= args.total_metric_calls:
            raise ValueError("warmup_metric_calls must be smaller than total_metric_calls for hybrid continuation.")
        return (
            {"override_max_metric_calls": args.warmup_metric_calls},
            {"override_max_metric_calls": args.total_metric_calls},
            f"warmup_validation_m{args.warmup_metric_calls}",
        )

    warmup_search_iterations = args.warmup_search_iterations or 50
    continuation_budgets = [
        args.total_metric_calls is not None,
        args.total_api_calls is not None,
        args.total_search_iterations is not None,
    ]
    if sum(continuation_budgets) != 1:
        raise ValueError("mem_llm_v3 requires exactly one of --total_metric_calls, --total_api_calls or --total_search_iterations.")
    warmup_budget = {"override_max_search_iterations": warmup_search_iterations}
    if args.total_api_calls is not None:
        warmup_budget["api_call_hard_limit"] = args.total_api_calls

    if args.total_metric_calls is not None:
        continuation_budget = {"override_max_metric_calls": args.total_metric_calls}
    elif args.total_api_calls is not None:
        continuation_budget = {"override_max_total_api_calls": args.total_api_calls}
    else:
        if args.total_search_iterations <= warmup_search_iterations:
            raise ValueError("total_search_iterations must be larger than warmup_search_iterations.")
        continuation_budget = {"override_max_search_iterations": args.total_search_iterations}
    return warmup_budget, continuation_budget, f"warmup_validation_i{warmup_search_iterations}"


def main() -> int:
    args = parse_args()
    warmup_budget_kwargs, continuation_budget_kwargs, warmup_budget_label = validate_budget_args(args)
    warmup_setting_name = f"{args.setting_name}__{warmup_budget_label}"
    validation_sidecar_judge_alignment = args.memory_protocol_version in {"mem_llm_v2", "mem_llm_v3"}
    lm_name = args.lm_config["name"]
    warmup_run_dir = run_dir_for(args.seed, args.benchmark_name, args.prog_name, args.optim_name, lm_name, warmup_setting_name)
    final_run_dir = run_dir_for(args.seed, args.benchmark_name, args.prog_name, args.optim_name, lm_name, args.setting_name)

    common_kwargs = {
        "dry_run": args.dry_run,
        "bm_idx": args.bm_idx,
        "benchmark_name": args.benchmark_name,
        "num_threads": args.num_threads,
        "program_idx": args.program_idx,
        "prog_name": args.prog_name,
        "opt_idx": args.opt_idx,
        "optim_name": args.optim_name,
        "lm_config": args.lm_config,
        "use_cache_from_opt": args.use_cache_from_opt,
        "seed": args.seed,
        "judge_lm_config": args.judge_lm_config,
        "judge_memory_top_k": args.judge_memory_top_k,
        "judge_memory_same_predictor_only": args.judge_memory_same_predictor_only,
        "resume_incomplete": args.resume_incomplete,
        "validation_sampling_mode": args.validation_sampling_mode,
        "smoke_train_size": args.smoke_train_size,
        "smoke_val_size": args.smoke_val_size,
        "smoke_test_size": args.smoke_test_size,
    }

    run_experiment_and_write_results(
        setting_name=warmup_setting_name,
        retained_validation_fraction=args.retained_validation_fraction,
        selection_mode="validation",
        validation_sidecar_judge_alignment=validation_sidecar_judge_alignment,
        always_validate_for_teacher_memory=args.memory_protocol_version == "mem_llm_v3",
        skip_final_evaluation=True,
        **warmup_budget_kwargs,
        **common_kwargs,
    )
    warmup_summary = load_json_from_path(warmup_run_dir / "seed_summary.json")
    warmup_actual_metric_calls = warmup_summary.get("actual_metric_calls", 0) or 0
    if args.total_metric_calls is not None and warmup_actual_metric_calls >= args.total_metric_calls:
        raise ValueError(
            f"Warmup already consumed {warmup_actual_metric_calls} metric calls, which reaches/exceeds total_metric_calls={args.total_metric_calls}."
        )
    learned_guide_info = {}
    if args.memory_protocol_version == "mem_llm_v3":
        learned_guide_info = write_learned_judge_guide_artifacts(
            run_dir=warmup_run_dir,
            max_cases=args.judge_memory_top_k,
        )

    phase_manifest = prepare_continuation_dir(
        warmup_run_dir=warmup_run_dir,
        final_run_dir=final_run_dir,
        warmup_setting_name=warmup_setting_name,
        memory_protocol_version=args.memory_protocol_version,
    )
    final_learned_guide_path = None
    if args.memory_protocol_version == "mem_llm_v3":
        final_learned_guide_path = str(final_run_dir / "judge_prompt_lessons.md")
        learned_guide_info.update(
            {
                "learned_judge_guide_path": final_learned_guide_path,
                "learned_judge_guide_json_path": str(final_run_dir / "judge_prompt_lessons.json"),
            }
        )
    if "override_max_total_api_calls" in continuation_budget_kwargs:
        warmup_search_api_calls = (warmup_summary.get("search_total_api_calls") or (
            (warmup_summary.get("optimizer_api_calls", 0) or 0)
            + (warmup_summary.get("judge_api_calls", 0) or 0)
        ))
        remaining_api_calls = continuation_budget_kwargs["override_max_total_api_calls"] - warmup_search_api_calls
        if remaining_api_calls <= 0:
            raise ValueError(
                f"Warmup consumed {warmup_search_api_calls} optimizer+judge API calls, exceeding total_api_calls={args.total_api_calls}."
            )
        continuation_budget_kwargs = {"override_max_total_api_calls": int(remaining_api_calls)}
    phase_manifest.update(
        {
            "final_setting_name": args.setting_name,
            "memory_protocol_version": args.memory_protocol_version,
            "warmup_budget": warmup_budget_kwargs,
            "warmup_actual_metric_calls": warmup_actual_metric_calls,
            "continuation_budget": continuation_budget_kwargs,
            "total_target_metric_calls": args.total_metric_calls,
            "total_target_api_calls": args.total_api_calls,
            "total_target_search_iterations": args.total_search_iterations,
            "retained_validation_fraction_in_warmup": args.retained_validation_fraction,
            "warmup_skipped_final_evaluation": True,
            "validation_sidecar_judge_alignment": validation_sidecar_judge_alignment,
            **learned_guide_info,
        }
    )
    write_json_to_path(final_run_dir / "hybrid_phase_manifest.json", phase_manifest)

    run_experiment_and_write_results(
        setting_name=args.setting_name,
        retained_validation_fraction=args.retained_validation_fraction,
        selection_mode="llm_judge",
        judge_learned_guide_path=final_learned_guide_path,
        judge_strict_learned_guide=args.memory_protocol_version == "mem_llm_v3",
        validation_sidecar_judge_alignment=False,
        skip_final_evaluation=False,
        **continuation_budget_kwargs,
        **common_kwargs,
    )
    final_summary = load_json_from_path(final_run_dir / "seed_summary.json")
    combined_summary = build_combined_summary(
        final_setting_name=args.setting_name,
        memory_protocol_version=args.memory_protocol_version,
        warmup_summary=warmup_summary,
        final_summary=final_summary,
        phase_manifest=phase_manifest,
    )
    write_json_to_path(final_run_dir / "hybrid_combined_summary.json", combined_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
