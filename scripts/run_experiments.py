import argparse
import copy
import math
import os
import os
import sys
import time
import json
import traceback
import random
import secrets
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
VENDORED_DSPY_ROOT = SRC_ROOT / "gepa_artifact" / "utils" / "dspy"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if VENDORED_DSPY_ROOT.exists() and str(VENDORED_DSPY_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_DSPY_ROOT))

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:
    def dotenv_values(_path=None):
        return {}

    def load_dotenv(_path=None, override=False):
        return False

from gepa_artifact.utils.capture_stream_logger import Logger

if __package__ in {None, ""}:
    from scripts.experiment_configs import BASE_EXPERIMENT_DIR, get_benchmarks, get_optimizers, get_max_invocations
else:
    from .experiment_configs import BASE_EXPERIMENT_DIR, get_benchmarks, get_optimizers, get_max_invocations

VALIDATION_SELECTION_MODE = "validation"
FEEDBACK_ONLY_LLM_JUDGE_MODE = "llm_judge"
LEGACY_SCORE_AWARE_LLM_JUDGE_MODE = "llm_judge_score_aware"
VALIDATION_LLM_JUDGE_COMBINED_MODE = "validation_llm_judge_combined"
LLM_JUDGE_SELECTION_MODES = {
    FEEDBACK_ONLY_LLM_JUDGE_MODE,
    LEGACY_SCORE_AWARE_LLM_JUDGE_MODE,
    VALIDATION_LLM_JUDGE_COMBINED_MODE,
}


def selection_mode_uses_judge(selection_mode: str) -> bool:
    return selection_mode in LLM_JUDGE_SELECTION_MODES


def selection_mode_uses_retained_validation(selection_mode: str) -> bool:
    return selection_mode in {
        VALIDATION_SELECTION_MODE,
        VALIDATION_LLM_JUDGE_COMBINED_MODE,
    }


def run_uses_sidecar_judge(selection_mode: str, validation_sidecar_judge_alignment: bool = False) -> bool:
    return selection_mode_uses_judge(selection_mode) or validation_sidecar_judge_alignment


def selection_mode_uses_formal_validation(selection_mode: str) -> bool:
    return selection_mode_uses_retained_validation(selection_mode)

def write_evaluation_result_to_path(evaluation_result, file_path):
    os.makedirs(file_path, exist_ok=True)
    file_name = f"evaluation_result"
    if evaluation_result.optimizer:
        optimizer_header = "optimizer,optimizer_cost,optimizer_input_tokens,optimizer_output_tokens"
        optimizer_values = (
            f"{evaluation_result.optimizer},{evaluation_result.optimizer_cost},"
            f"{evaluation_result.optimizer_input_tokens},{evaluation_result.optimizer_output_tokens},"
        )
    else:
        optimizer_header = ""
        optimizer_values = ""
    with open(os.path.join(file_path, f"{file_name}.txt"), "w") as f:
        f.write(f"score,cost,input_tokens,output_tokens,{optimizer_header}\n")
        f.write(
            f"{evaluation_result.score},{evaluation_result.cost},{evaluation_result.input_tokens},"
            f"{evaluation_result.output_tokens},{optimizer_values}\n"
        )
    if evaluation_result.optimizer:
        evaluation_result.optimized_program.save(
            os.path.join(file_path, f"optimized_program"),
            save_program=True
        )
    if evaluation_result.optimizer_program_scores:
        with open(
            os.path.join(file_path, f"{file_name}_optimizer_score.txt"), "w"
        ) as f:
            f.write(",".join(evaluation_result.optimizer_program_scores))

def write_json_to_path(file_path, payload):
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_json_from_path(file_path, default=None):
    if not os.path.exists(file_path):
        return {} if default is None else default
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def coerce_evaluation_score(score_or_result):
    score = getattr(score_or_result, 'score', score_or_result)
    try:
        return float(score)
    except (TypeError, ValueError):
        return score


def get_lm_usage_stats(lm) -> dict[str, float | int | str | None]:
    if lm is None:
        return {
            "model": None,
            "cost": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "api_calls": 0,
        }

    cost = 0
    input_tokens = 0
    output_tokens = 0
    for trace in lm.history:
        cost += trace.get("cost", None) or 0
        input_tokens += trace.get("usage", 0).get("prompt_tokens", 0)
        output_tokens += trace.get("usage", 0).get("completion_tokens", 0)

    return {
        "model": getattr(lm, "model", None),
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "api_calls": len(lm.history),
    }


def calculate_stats(lm_or_lms) -> tuple[float, int, int]:
    if lm_or_lms is None:
        return 0, 0, 0

    lms = lm_or_lms if isinstance(lm_or_lms, (list, tuple, set)) else [lm_or_lms]
    seen_ids = set()
    cost = 0
    input_tokens = 0
    output_tokens = 0
    for lm in lms:
        if lm is None or id(lm) in seen_ids:
            continue
        seen_ids.add(id(lm))
        usage = get_lm_usage_stats(lm)
        cost += usage["cost"]
        input_tokens += usage["input_tokens"]
        output_tokens += usage["output_tokens"]

    return cost, input_tokens, output_tokens


def resolve_api_key_env_vars(lm_config):
    if lm_config is None:
        return None
    resolved = copy.deepcopy(lm_config)
    if 'api_key' in resolved and isinstance(resolved['api_key'], str) and resolved['api_key'].startswith('env:'):
        env_var = resolved['api_key'].split(':', 1)[1]
        if env_var in os.environ:
            resolved['api_key'] = os.environ[env_var]
        else:
            raise ValueError(f"Environment variable {env_var} not found. Please set it before running the script. It is required for the LM configuration.")
    return resolved


def bootstrap_openai_compatible_env():
    dotenv_path = PROJECT_ROOT / ".env"
    dotenv_payload = dotenv_values(dotenv_path) if dotenv_path.exists() else {}
    load_dotenv(dotenv_path, override=False)

    pythonpath_roots = []
    vendor_dir = PROJECT_ROOT / ".vendor"
    if vendor_dir.exists():
        pythonpath_roots.append(vendor_dir)
    if VENDORED_DSPY_ROOT.exists():
        pythonpath_roots.append(VENDORED_DSPY_ROOT)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
    for root in pythonpath_roots:
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        if root_str not in pythonpath_entries:
            pythonpath_entries.insert(0, root_str)
    if pythonpath_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    ssl_keylogfile = os.environ.get("SSLKEYLOGFILE")
    if ssl_keylogfile:
        try:
            ssl_keylog_path = Path(ssl_keylogfile)
            ssl_keylog_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ssl_keylog_path, "a", encoding="utf-8"):
                pass
        except OSError:
            os.environ.pop("SSLKEYLOGFILE", None)

    placeholder_api_keys = {"your_openai_api_key", "[your_openai_api_key]"}
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("PAPILLON_API_KEY")
    dotenv_api_key = (
        dotenv_payload.get("OPENAI_API_KEY")
        or dotenv_payload.get("PAPILLON_API_KEY")
    )
    if api_key and api_key.strip() in placeholder_api_keys and dotenv_api_key:
        api_key = dotenv_api_key
    if api_key:
        api_key = api_key.strip()
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ.setdefault("PAPILLON_API_KEY", api_key)

    api_base = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("PAPILLON_API_BASE")
    )
    if api_base:
        api_base = api_base.strip()
        os.environ["OPENAI_BASE_URL"] = api_base
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ.setdefault("PAPILLON_API_BASE", api_base)

    globals()["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
    globals()["wandb_api_key"] = os.environ.get("WANDB_API_KEY")


def prune_jsonl_records_by_iteration(file_path, max_iteration):
    if not os.path.exists(file_path):
        return
    kept_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            iteration = payload.get("iteration")
            if iteration is None or iteration <= max_iteration:
                kept_lines.append(stripped)
    with open(file_path, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")


def prepare_resume_sidecars(run_dir):
    from gepa_artifact.gepa.gepa_utils import GEPAState

    gepa_state = GEPAState.load(run_dir)
    max_iteration = gepa_state.i + 1
    for filename in (
        "iteration_summary.jsonl",
        "judge_decisions.jsonl",
        "metric_call_checkpoints.jsonl",
        "judge_memory_retrievals.jsonl",
        "judge_memory_bank.jsonl",
        "judge_alignment_memory_bank.jsonl",
    ):
        prune_jsonl_records_by_iteration(os.path.join(run_dir, filename), max_iteration)
    return {
        "resumed_from_existing_state": True,
        "resume_state_iteration": max_iteration,
        "resume_state_metric_calls": gepa_state.total_num_evals,
        "resume_state_num_candidates": len(gepa_state.program_candidates),
    }


def apply_retained_validation_fraction(
    benchmark,
    retained_validation_fraction,
    selection_mode,
    seed,
    validation_sampling_mode="fixed",
    validation_subset_seed=None,
    existing_subset_payload=None,
):
    if validation_sampling_mode != "fixed":
        raise ValueError(
            "validation_sampling_mode='random_per_iteration' has been removed. "
            "Use the default fixed mode, which randomly selects the retained validation subset once per run."
        )

    original_val_size = len(benchmark.val_set)
    existing_retained_positions = None
    selection_seed_source = "generated"
    if existing_subset_payload:
        if existing_subset_payload.get("sampling_mode") != "fixed":
            raise ValueError(
                f"Cannot reuse validation subset with sampling_mode={existing_subset_payload.get('sampling_mode')!r}; "
                "only fixed validation subsets are supported."
            )
        existing_retained_positions = existing_subset_payload.get("retained_positions")
        validation_subset_seed = existing_subset_payload.get("selection_seed", validation_subset_seed)
        selection_seed_source = "existing_run"

    if validation_subset_seed is None and not existing_retained_positions:
        validation_subset_seed = secrets.randbits(32)
    elif selection_seed_source != "existing_run":
        selection_seed_source = "provided"

    payload = {
        "selection_mode": selection_mode,
        "retained_validation_fraction": retained_validation_fraction,
        "original_val_size": original_val_size,
        "selection_seed": validation_subset_seed,
        "selection_seed_source": selection_seed_source,
    }

    if selection_mode_uses_judge(selection_mode) and not selection_mode_uses_retained_validation(selection_mode):
        payload.update({
            "applied": False,
            "reason": "formal validation set is unused during llm_judge selection",
            "retained_val_size": original_val_size,
            "retained_positions": list(range(original_val_size)),
        })
        return payload

    if retained_validation_fraction <= 0 or retained_validation_fraction > 100:
        raise ValueError(f"retained_validation_fraction must be in (0, 100], got {retained_validation_fraction}")

    if original_val_size == 0:
        payload.update({
            "applied": False,
            "reason": "benchmark has empty validation set",
            "retained_val_size": 0,
            "retained_positions": [],
        })
        return payload

    keep_count = original_val_size if retained_validation_fraction >= 100 else max(1, math.ceil(original_val_size * retained_validation_fraction / 100.0))

    if existing_retained_positions is not None:
        retained_positions = list(existing_retained_positions)
        if len(retained_positions) != keep_count:
            raise ValueError(
                f"Existing validation subset has {len(retained_positions)} examples, "
                f"but retained_validation_fraction={retained_validation_fraction} requires {keep_count}."
            )
        if any(pos < 0 or pos >= original_val_size for pos in retained_positions):
            raise ValueError("Existing validation subset contains positions outside the current validation split.")
        retained_positions = sorted(retained_positions)
    elif retained_validation_fraction >= 100:
        retained_positions = list(range(original_val_size))
    else:
        selection_rng = random.Random(validation_subset_seed)
        candidate_positions = list(range(original_val_size))
        selection_rng.shuffle(candidate_positions)
        retained_positions = sorted(candidate_positions[:keep_count])

    benchmark.val_set = [benchmark.val_set[i] for i in retained_positions]
    payload.update({
        "applied": keep_count != original_val_size,
        "sampling_mode": "fixed",
        "retained_val_size": len(benchmark.val_set),
        "retained_positions": retained_positions,
    })
    return payload


def apply_smoke_subset(benchmark, seed, smoke_train_size=None, smoke_val_size=None, smoke_test_size=None):
    requested_sizes = {
        "train": smoke_train_size,
        "val": smoke_val_size,
        "test": smoke_test_size,
    }
    payload = {
        "selection_seed": seed + 12347,
        "requested_sizes": requested_sizes,
        "original_sizes": {
            "train": len(benchmark.train_set),
            "val": len(benchmark.val_set),
            "test": len(benchmark.test_set),
        },
        "applied": False,
        "retained_positions": {},
    }

    split_offsets = {"train": 11, "val": 23, "test": 37}
    for split_name, requested_size in requested_sizes.items():
        dataset = getattr(benchmark, f"{split_name}_set")
        original_size = len(dataset)
        if requested_size is None:
            payload["retained_positions"][split_name] = list(range(original_size))
            continue
        if requested_size <= 0:
            raise ValueError(f"smoke_{split_name}_size must be positive when provided, got {requested_size}")
        if requested_size >= original_size:
            payload["retained_positions"][split_name] = list(range(original_size))
            continue

        selection_rng = random.Random(seed + 12347 + split_offsets[split_name])
        candidate_positions = list(range(original_size))
        selection_rng.shuffle(candidate_positions)
        retained_positions = sorted(candidate_positions[:requested_size])
        setattr(benchmark, f"{split_name}_set", [dataset[i] for i in retained_positions])
        payload["retained_positions"][split_name] = retained_positions
        payload["applied"] = True

    payload["retained_sizes"] = {
        "train": len(benchmark.train_set),
        "val": len(benchmark.val_set),
        "test": len(benchmark.test_set),
    }
    return payload


def env_flag_is_true(name: str, default: bool = True) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_effective_launch_arbor(optimizer_config, lm_config, optim_name: str):
    requested = bool(
        optimizer_config is not None
        and "launch_arbor" in optimizer_config.langProBe_configs
        and optimizer_config.langProBe_configs["launch_arbor"]
    )
    if not requested:
        return False, False, None

    if not env_flag_is_true("GEPA_ENABLE_LAUNCH_ARBOR", default=True):
        return False, True, "disabled by GEPA_ENABLE_LAUNCH_ARBOR=0"

    reasons = []
    num_gpus = None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        num_gpus = len([line for line in output.splitlines() if line.strip()])
    except Exception:
        pass

    if num_gpus is None:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except Exception as exc:
            return False, True, f"failed to inspect GPU count for ArborRunner: {exc}"

    if "GRPO" in optim_name:
        arbor_config_path = PROJECT_ROOT / "utils" / "arbor" / "arbor_train.yaml"
    elif num_gpus == 4:
        arbor_config_path = PROJECT_ROOT / "utils" / "arbor" / "arbor_inference.yaml"
    elif num_gpus == 2:
        arbor_config_path = PROJECT_ROOT / "utils" / "arbor" / "arbor_inference_2_gpus.yaml"
    else:
        reasons.append(f"unsupported GPU count {num_gpus} for ArborRunner")
        arbor_config_path = None

    if arbor_config_path is not None and not arbor_config_path.exists():
        reasons.append(f"missing Arbor config file: {arbor_config_path}")

    env_sh_path = PROJECT_ROOT / "env.sh"
    if not env_sh_path.exists():
        reasons.append(f"missing Arbor environment bootstrap file: {env_sh_path}")

    api_base = (lm_config or {}).get("api_base")
    if "{portnum}" not in str(api_base):
        reasons.append("lm_config.api_base must contain '{portnum}' when launch_arbor is enabled")

    if reasons:
        return False, True, "; ".join(reasons)

    return True, True, None


def extract_gepa_run_summary(run_dir):
    gepa_state_path = os.path.join(run_dir, "gepa_state.bin")
    if not os.path.exists(gepa_state_path):
        return {}

    from gepa_artifact.gepa.gepa_utils import GEPAState

    gepa_state = GEPAState.load(run_dir)
    accepted_updates = []
    for trace in gepa_state.full_program_trace:
        if "new_program_idx" not in trace:
            continue
        judge_decision = trace.get("judge_decision", {})
        accepted_updates.append({
            "iteration": trace["i"] + 1,
            "selected_program_candidate": trace.get("selected_program_candidate"),
            "predictor_name_to_update": trace.get("predictor_name_to_update"),
            "new_program_idx": trace.get("new_program_idx"),
            "judge_preferred_prompt": judge_decision.get("preferred_prompt"),
            "judge_confidence": judge_decision.get("confidence"),
            "selection_surrogate_score": trace.get("selection_surrogate_score"),
        })

    return {
        "actual_search_iterations": gepa_state.i + 1,
        "actual_metric_calls": gepa_state.total_num_evals,
        "actual_num_evals_per_trainval_instance": gepa_state.total_num_evals_per_trainval_instance,
        "num_candidates": len(gepa_state.program_candidates),
        "accepted_updates": len(accepted_updates),
        "best_tracked_score": max(gepa_state.per_program_tracked_scores),
        "best_program_idx": gepa_state.per_program_tracked_scores.index(max(gepa_state.per_program_tracked_scores)),
        "validation_input_tokens": getattr(gepa_state, "validation_input_tokens", 0),
        "validation_output_tokens": getattr(gepa_state, "validation_output_tokens", 0),
        "validation_api_calls": getattr(gepa_state, "validation_api_calls", 0),
        "validation_tokens": getattr(gepa_state, "validation_input_tokens", 0) + getattr(gepa_state, "validation_output_tokens", 0),
        "minibatch_input_tokens": getattr(gepa_state, "minibatch_input_tokens", 0),
        "minibatch_output_tokens": getattr(gepa_state, "minibatch_output_tokens", 0),
        "minibatch_api_calls": getattr(gepa_state, "minibatch_api_calls", 0),
        "minibatch_tokens": getattr(gepa_state, "minibatch_input_tokens", 0) + getattr(gepa_state, "minibatch_output_tokens", 0),
        "accepted_update_records": accepted_updates,
    }


def load_last_jsonl_record(file_path):
    if not os.path.exists(file_path):
        return {}
    last_record = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                last_record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
    return last_record


def count_jsonl_records(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def parse_evaluation_result_file(file_path):
    if not os.path.exists(file_path):
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    if len(rows) < 2:
        return {}

    headers = rows[0].split(",")
    values = rows[1].split(",")
    payload = {}
    int_fields = {"input_tokens", "output_tokens", "optimizer_input_tokens", "optimizer_output_tokens"}
    float_fields = {"score", "cost", "optimizer_cost"}
    for key, raw_value in zip(headers, values):
        if raw_value == "":
            continue
        if key in int_fields:
            payload[key] = int(float(raw_value))
        elif key in float_fields:
            payload[key] = float(raw_value)
        else:
            payload[key] = raw_value
    return payload


def build_run_summaries(
    *,
    runs_dir,
    benchmark,
    benchmark_name,
    prog_name,
    optim_name,
    lm_name,
    lm_config,
    judge_lm_config,
    selection_mode,
    validation_sidecar_judge_alignment,
    always_validate_for_teacher_memory=False,
    retained_validation_fraction,
    seed,
    setting_name,
    smoke_subset_payload,
    validation_subset_payload,
    phase_wall_clock,
    run_name,
    resume_metadata,
    skip_final_evaluation=False,
    eval_results=None,
    optimizer_lm_usage=None,
    judge_lm_usage=None,
    evaluation_lm_usage=None,
    combined_score_mode="normalized",
    combined_validation_weight=1.0,
    combined_judge_weight=1.0,
    combined_validation_score_scale=100.0,
    combined_min_surrogate_gain=0.01,
):
    gepa_run_summary = extract_gepa_run_summary(runs_dir)
    accepted_update_records = gepa_run_summary.pop("accepted_update_records", [])

    iteration_tail = load_last_jsonl_record(os.path.join(runs_dir, "iteration_summary.jsonl"))
    evaluation_result_payload = parse_evaluation_result_file(
        os.path.join(runs_dir, "evaluation_results", "evaluation_result.txt")
    )

    resolved_lm_model = lm_config.get("new_model_name", lm_config.get("model"))
    resolved_judge_model = None
    if run_uses_sidecar_judge(selection_mode, validation_sidecar_judge_alignment):
        judge_source_config = judge_lm_config or lm_config
        resolved_judge_model = judge_source_config.get("new_model_name", judge_source_config.get("model"))

    if optimizer_lm_usage is None:
        optimizer_lm_usage = {
            "model": resolved_lm_model,
            "cost": evaluation_result_payload.get("optimizer_cost", 0),
            "input_tokens": iteration_tail.get("optimizer_input_tokens", evaluation_result_payload.get("optimizer_input_tokens", 0)),
            "output_tokens": iteration_tail.get("optimizer_output_tokens", evaluation_result_payload.get("optimizer_output_tokens", 0)),
            "api_calls": iteration_tail.get("optimizer_api_calls", 0),
        }

    if judge_lm_usage is None:
        judge_lm_usage = {
            "model": resolved_judge_model,
            "cost": 0,
            "input_tokens": iteration_tail.get("judge_input_tokens", 0),
            "output_tokens": iteration_tail.get("judge_output_tokens", 0),
            "api_calls": iteration_tail.get("judge_api_calls", 0),
        }

    if evaluation_lm_usage is None:
        evaluation_lm_usage = {
            "model": resolved_lm_model,
            "cost": evaluation_result_payload.get("cost", 0),
            "input_tokens": evaluation_result_payload.get("input_tokens", 0),
            "output_tokens": evaluation_result_payload.get("output_tokens", 0),
            "api_calls": count_jsonl_records(os.path.join(runs_dir, "metric_logs", "test.jsonl")),
        }

    if eval_results is None:
        final_test_score = evaluation_result_payload.get("score")
        optimizer_cost = evaluation_result_payload.get("optimizer_cost", 0)
        optimizer_input_tokens = evaluation_result_payload.get("optimizer_input_tokens", optimizer_lm_usage["input_tokens"])
        optimizer_output_tokens = evaluation_result_payload.get("optimizer_output_tokens", optimizer_lm_usage["output_tokens"])
        evaluation_cost = evaluation_result_payload.get("cost", 0)
        evaluation_input_tokens = evaluation_result_payload.get("input_tokens", evaluation_lm_usage["input_tokens"])
        evaluation_output_tokens = evaluation_result_payload.get("output_tokens", evaluation_lm_usage["output_tokens"])
    else:
        final_test_score = eval_results.score
        optimizer_cost = eval_results.optimizer_cost
        optimizer_input_tokens = eval_results.optimizer_input_tokens
        optimizer_output_tokens = eval_results.optimizer_output_tokens
        evaluation_cost = eval_results.cost
        evaluation_input_tokens = eval_results.input_tokens
        evaluation_output_tokens = eval_results.output_tokens

    llm_usage_summary = {
        "optimizer_lm": optimizer_lm_usage,
        "judge_lm": judge_lm_usage,
        "evaluation_lm": evaluation_lm_usage,
        "validation_selection": {
            "input_tokens": gepa_run_summary.get("validation_input_tokens", 0),
            "output_tokens": gepa_run_summary.get("validation_output_tokens", 0),
            "total_tokens": gepa_run_summary.get("validation_tokens", 0),
            "api_calls": gepa_run_summary.get("validation_api_calls", 0),
        },
    }
    optimizer_total_tokens = (optimizer_lm_usage.get("input_tokens", 0) or 0) + (optimizer_lm_usage.get("output_tokens", 0) or 0)
    judge_total_tokens = (judge_lm_usage.get("input_tokens", 0) or 0) + (judge_lm_usage.get("output_tokens", 0) or 0)
    validation_tokens = gepa_run_summary.get("validation_tokens", 0) or 0
    minibatch_tokens = gepa_run_summary.get("minibatch_tokens", 0) or 0
    optimizer_overhead_tokens = max(0, optimizer_total_tokens - validation_tokens - minibatch_tokens)
    optimizer_overhead_api_calls = max(
        0,
        (optimizer_lm_usage.get("api_calls", 0) or 0)
        - (gepa_run_summary.get("validation_api_calls", 0) or 0)
        - (gepa_run_summary.get("minibatch_api_calls", 0) or 0),
    )
    cost_accounting = {
        "accounting_basis": "validation and minibatch are subsets of optimizer_lm; do not add them again to optimizer_total_tokens",
        "validation": llm_usage_summary["validation_selection"],
        "minibatch": {
            "input_tokens": gepa_run_summary.get("minibatch_input_tokens", 0),
            "output_tokens": gepa_run_summary.get("minibatch_output_tokens", 0),
            "total_tokens": minibatch_tokens,
            "api_calls": gepa_run_summary.get("minibatch_api_calls", 0),
        },
        "judge": {
            "input_tokens": judge_lm_usage.get("input_tokens", 0),
            "output_tokens": judge_lm_usage.get("output_tokens", 0),
            "total_tokens": judge_total_tokens,
            "api_calls": judge_lm_usage.get("api_calls", 0),
        },
        "optimizer_overhead": {
            "total_tokens": optimizer_overhead_tokens,
            "api_calls": optimizer_overhead_api_calls,
        },
        "optimization_control": {
            "total_tokens": optimizer_overhead_tokens + judge_total_tokens,
            "api_calls": optimizer_overhead_api_calls + (judge_lm_usage.get("api_calls", 0) or 0),
        },
        "search_total": {
            "total_tokens": optimizer_total_tokens + judge_total_tokens,
            "api_calls": (optimizer_lm_usage.get("api_calls", 0) or 0) + (judge_lm_usage.get("api_calls", 0) or 0),
        },
    }
    llm_usage_summary["cost_accounting"] = cost_accounting

    seed_summary = {
        "setting": setting_name,
        "setting_name": setting_name,
        "benchmark_name": benchmark_name,
        "program_name": prog_name,
        "optimizer_name": optim_name,
        "lm_name": lm_name,
        "selection_mode": selection_mode,
        "validation_sidecar_judge_alignment": validation_sidecar_judge_alignment,
        "always_validate_for_teacher_memory": always_validate_for_teacher_memory,
        "combined_score_mode": combined_score_mode,
        "combined_validation_weight": combined_validation_weight,
        "combined_judge_weight": combined_judge_weight,
        "combined_validation_score_scale": combined_validation_score_scale,
        "combined_min_surrogate_gain": combined_min_surrogate_gain,
        "retained_validation_fraction": retained_validation_fraction,
        "skip_final_evaluation": skip_final_evaluation,
        "seed": seed,
        "benchmark_subset": smoke_subset_payload,
        "train_size": len(benchmark.train_set),
        "val_size": len(benchmark.val_set),
        "test_size": len(benchmark.test_set),
        "final_test_score": final_test_score,
        "optimizer_cost": optimizer_cost,
        "optimizer_input_tokens": optimizer_input_tokens,
        "optimizer_output_tokens": optimizer_output_tokens,
        "evaluation_cost": evaluation_cost,
        "evaluation_input_tokens": evaluation_input_tokens,
        "evaluation_output_tokens": evaluation_output_tokens,
        "total_cost": (optimizer_cost or 0) + (evaluation_cost or 0),
        "total_input_tokens": (optimizer_input_tokens or 0) + (evaluation_input_tokens or 0),
        "total_output_tokens": (optimizer_output_tokens or 0) + (evaluation_output_tokens or 0),
        "total_tokens": (optimizer_input_tokens or 0) + (evaluation_input_tokens or 0) + (optimizer_output_tokens or 0) + (evaluation_output_tokens or 0),
        "total_api_calls": optimizer_lm_usage["api_calls"] + judge_lm_usage["api_calls"] + evaluation_lm_usage["api_calls"],
        "optimizer_api_calls": optimizer_lm_usage["api_calls"],
        "judge_api_calls": judge_lm_usage["api_calls"],
        "evaluation_api_calls": evaluation_lm_usage["api_calls"],
        "minibatch_tokens": minibatch_tokens,
        "minibatch_api_calls": gepa_run_summary.get("minibatch_api_calls", 0),
        "optimizer_overhead_tokens": optimizer_overhead_tokens,
        "optimizer_overhead_api_calls": optimizer_overhead_api_calls,
        "optimization_control_tokens": optimizer_overhead_tokens + judge_total_tokens,
        "optimization_control_api_calls": optimizer_overhead_api_calls + (judge_lm_usage.get("api_calls", 0) or 0),
        "search_total_tokens": optimizer_total_tokens + judge_total_tokens,
        "search_total_api_calls": (optimizer_lm_usage.get("api_calls", 0) or 0) + (judge_lm_usage.get("api_calls", 0) or 0),
        "phase_wall_clock_breakdown": phase_wall_clock,
        "total_time_seconds": phase_wall_clock.get("total_seconds", 0),
        "validation_subset": validation_subset_payload,
        "run_name": run_name,
        **resume_metadata,
        **gepa_run_summary,
    }
    return seed_summary, llm_usage_summary, accepted_update_records


def write_run_summaries(**kwargs):
    seed_summary, llm_usage_summary, accepted_update_records = build_run_summaries(**kwargs)
    runs_dir = kwargs["runs_dir"]
    write_json_to_path(os.path.join(runs_dir, "accepted_updates.json"), accepted_update_records)
    write_json_to_path(os.path.join(runs_dir, "llm_usage_summary.json"), llm_usage_summary)
    write_json_to_path(os.path.join(runs_dir, "seed_summary.json"), seed_summary)


def create_lm(lm_config):
    import dspy
    config = lm_config.copy()
    config['model'] = config.pop("new_model_name", config['model'])
    provider = None
    if "openai/arbor" in config['model']:
        from dspy.clients.lm_local_arbor import ArborProvider
        provider = ArborProvider()
    config = {k:v for k, v in config.items() if k != "name"}
    config.setdefault("max_tokens", 16384)  # override DSPy defaults only when caller did not specify one
    config.setdefault("num_retries", 0)
    if provider is not None and "provider" not in config:
        config["provider"] = provider
    return dspy.LM(**config)


def get_free_port() -> int:
    """
    Return a randomly selected free TCP port on localhost from a selection of 3-4 ports.
    """
    import random
    import socket
    ports = []
    for _ in range(random.randint(5, 10)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", 0))
                ports.append(s.getsockname()[1])
        except Exception as e:
            print(f"Error binding to port: {e}")
    return random.choice(ports)

def run_experiment_and_write_results_actual(
    bm_idx,
    benchmark_name,
    num_threads,
    program_idx,
    prog_name,
    opt_idx,
    optim_name,
    lm_config,
    dry_run=False,
    use_cache_from_opt=None,
    seed=0,
    setting_name=None,
    retained_validation_fraction=100.0,
    validation_sampling_mode="fixed",
    validation_subset_seed=None,
    selection_mode="validation",
    judge_lm_config=None,
    judge_learned_guide_path=None,
    judge_strict_learned_guide=False,
    judge_memory_top_k=3,
    judge_memory_same_predictor_only=True,
    validation_sidecar_judge_alignment=False,
    always_validate_for_teacher_memory=False,
    combined_score_mode="normalized",
    combined_validation_weight=1.0,
    combined_judge_weight=1.0,
    combined_validation_score_scale=100.0,
    combined_min_surrogate_gain=0.01,
    override_max_metric_calls=None,
    override_max_total_api_calls=None,
    override_max_total_search_tokens=None,
    api_call_hard_limit=None,
    override_max_search_iterations=None,
    skip_final_evaluation=False,
    resume_incomplete=True,
    smoke_train_size=None,
    smoke_val_size=None,
    smoke_test_size=None,
    log_all_io=None,
):
    bootstrap_openai_compatible_env()
    if benchmark_name == "Papillon":
        os.environ.setdefault("GEPA_PAPILLON_AUX_MODEL", "openai/qwen3-8b")
    base_experiment_dir = BASE_EXPERIMENT_DIR
    lm_config = resolve_api_key_env_vars(copy.deepcopy(lm_config))
    judge_lm_config = resolve_api_key_env_vars(copy.deepcopy(judge_lm_config))
    lm_name = lm_config["name"]
    print(f"Running {benchmark_name} with {prog_name} and {optim_name} on {lm_name}")
    runs_dir_basepath = os.path.join(base_experiment_dir, "experiment_runs", f"seed_{seed}")
    use_wandb = env_flag_is_true("GEPA_USE_WANDB", default=True)
    run_name = f"{benchmark_name}_{prog_name}_{optim_name}_{lm_name}"
    if setting_name:
        run_name = f"{run_name}__{setting_name}"
    runs_dir = os.path.join(runs_dir_basepath, run_name)
    log_all_io_effective = (
        env_flag_is_true("GEPA_LOG_ALL_IO", default=False)
        or env_flag_is_true("GEPA_LM_IO_TRACE_ENABLED", default=False)
    ) if log_all_io is None else bool(log_all_io)
    from gepa_artifact.utils.lm_io_trace import configure_lm_io_trace, lm_trace_context

    lm_io_trace_path = configure_lm_io_trace(
        run_dir=runs_dir,
        run_id=f"{run_name}__seed_{seed}",
        enabled=log_all_io_effective,
    )

    #######################
    # Cache Setup:
    # use_cache_from_opt is used to ensure consistency
    #######################
    cache_dir = os.path.join(base_experiment_dir, "experiment_cache_dirs", f"seed_{seed}", run_name)
    if use_cache_from_opt is None:
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_source_run_name = f"{benchmark_name}_{prog_name}_{use_cache_from_opt}_{lm_name}"
        cache_source_cache_subdir = os.path.join("experiment_cache_dirs", f"seed_{seed}", cache_source_run_name)
        cache_source_cache_dir = os.path.join(base_experiment_dir, "experiment_cache_dirs", f"seed_{seed}", cache_source_run_name)
        if not os.path.exists(cache_dir):
            cache_source_run_dir = os.path.join(base_experiment_dir, "experiment_runs", f"seed_{seed}", cache_source_run_name)
            # Now, we will wait indefinitely till the source run has evaluation results ready
            print(f"Waiting for {cache_source_run_dir} to have evaluation results ready...")
            while not os.path.exists(os.path.join(cache_source_run_dir, "evaluation_results")):
                time.sleep(100)
            print(f"Found evaluation results for {cache_source_run_name} in {cache_source_run_dir}.")
            print(f"Copying cache from {cache_source_cache_dir} to {cache_dir}...")
            import shutil
            shutil.copytree(cache_source_cache_dir, cache_dir)
            # Write a small token marker file to indicate this was copied from the source run
            with open(os.path.join(cache_dir, "cache_from_source_run.txt"), "w") as f:
                f.write(f"Copied from {cache_source_cache_dir}")
            print(f"Copied cache from {cache_source_cache_dir} to {cache_dir}. Continuing the current run with the cache from {cache_source_run_name}...")
            assert os.path.exists(os.path.join(cache_dir, ".dspy_cache"))
        else:
            assert os.path.exists(os.path.join(cache_dir, "cache_from_source_run.txt"))
            with open(os.path.join(cache_dir, "cache_from_source_run.txt"), "r") as f:
                assert cache_source_cache_subdir in f.read() # == f"Copied from {cache_source_cache_dir}"

    dspy_cachedir = os.path.join(cache_dir, ".dspy_cache")
    os.environ["DSPY_CACHEDIR"] = dspy_cachedir
    os.environ["DSP_CACHEDIR"] = dspy_cachedir
    os.environ["DSPY_NOTEBOOK_CACHEDIR"] = dspy_cachedir
    os.environ["DSP_NOTEBOOK_CACHEDIR"] = dspy_cachedir
    import dspy

    from gepa_artifact.benchmarks.benchmark import EvaluationResult
    from gepa_artifact.utils.metric_logger import MetricWithLogger, CounterWithLock
    from gepa_artifact.utils.json_default_encoder import json_encoder

    metric_lm_name = lm_config.get("metric_lm_name", lm_config["name"])
    metric_lm = lm_config.get("model", None)

    adapter = dspy.settings.adapter # if "qwen" not in lm_name else XMLAdapter()
    evalsetname = "testset"

    #######################
    # Obtain the benchmark, program and optimizer to execute
    #######################
    benchmark_metas, optimizers = get_benchmarks([benchmark_name]), get_optimizers()
    assert len(benchmark_metas) == 1, f"Expected exactly one benchmark meta for {benchmark_name}, found {len(benchmark_metas)}"
    benchmark_meta = benchmark_metas[0]
    program = benchmark_meta.program[program_idx]
    optimizer_config = copy.deepcopy(optimizers[opt_idx][1])
    benchmark = benchmark_meta.benchmark()
    launch_arbor_effective, launch_arbor_requested, launch_arbor_disable_reason = resolve_effective_launch_arbor(
        optimizer_config,
        lm_config,
        optim_name,
    )
    if optimizer_config is not None and "launch_arbor" in optimizer_config.langProBe_configs:
        optimizer_config.langProBe_configs["launch_arbor"] = launch_arbor_effective

    if use_cache_from_opt is not None:
        assert "use_cache_from_opt" in optimizer_config.langProBe_configs
        assert optimizer_config.langProBe_configs["use_cache_from_opt"] == use_cache_from_opt

    if seed != 0:
        # Shuffle the examples
        print("Shuffling the data splits: train and val")
        train_size = len(benchmark.train_set)
        combined_train_val = benchmark.train_set + benchmark.val_set
        random.Random(seed).shuffle(combined_train_val)
        benchmark.train_set = combined_train_val[:train_size]
        benchmark.val_set = combined_train_val[train_size:]

    existing_validation_subset_payload = None
    existing_validation_subset_path = os.path.join(runs_dir, "validation_subset_ids.json")
    if os.path.exists(existing_validation_subset_path):
        existing_payload = load_json_from_path(existing_validation_subset_path, default={})
        if existing_payload.get("sampling_mode") == "fixed":
            existing_validation_subset_payload = existing_payload
        elif resume_incomplete and os.path.exists(os.path.join(runs_dir, "gepa_state.bin")):
            raise ValueError(
                f"{runs_dir} was created with validation_sampling_mode={existing_payload.get('sampling_mode')!r}. "
                "Per-iteration validation resampling has been removed, so this run cannot be resumed safely. "
                "Move the old run directory aside and start a new fixed-subset run."
            )

    smoke_subset_payload = apply_smoke_subset(
        benchmark=benchmark,
        seed=seed,
        smoke_train_size=smoke_train_size,
        smoke_val_size=smoke_val_size,
        smoke_test_size=smoke_test_size,
    )
    validation_subset_payload = apply_retained_validation_fraction(
        benchmark=benchmark,
        retained_validation_fraction=retained_validation_fraction,
        selection_mode=selection_mode,
        seed=seed,
        validation_sampling_mode=validation_sampling_mode,
        validation_subset_seed=validation_subset_seed,
        existing_subset_payload=existing_validation_subset_payload,
    )

    assert benchmark_name == (benchmark_meta.name or benchmark.__class__.__name__)
    assert num_threads <= (benchmark_meta.num_threads or os.cpu_count())
    assert prog_name == getattr(program, "_name", program.__class__.__name__)
    assert optim_name == optimizers[opt_idx][0]
    if optimizers[opt_idx][1] is not None:
        assert optimizers[opt_idx][1].name == optim_name

    if optimizer_config is not None and "run_constraints" in optimizer_config.langProBe_configs:
        run_constraints = optimizer_config.langProBe_configs["run_constraints"]
        if "benchmark_name" in run_constraints and benchmark_name not in run_constraints["benchmark_name"]:
            print(f"Skipping {benchmark_name} because it does not match the run constraints {run_constraints}")
            return

    if getattr(program, "run_constraints", None) is not None:
        run_constraints = program.run_constraints
        if "benchmark_name" in run_constraints and benchmark_name not in run_constraints["benchmark_name"]:
            print(f"Skipping {benchmark_name} because it does not match the run constraints {run_constraints}")
            return
        if "model_name" in run_constraints and lm_name not in run_constraints["model_name"]:
            print(f"Skipping {benchmark_name} because it does not match the run constraints {run_constraints}")
            return
        if "optimizer_name" in run_constraints and optim_name not in run_constraints["optimizer_name"]:
            print(f"Skipping {benchmark_name} because it does not match the run constraints {run_constraints}")
            return

    #######################
    # Check if this experiment has already executed successfully, if so, skip
    #######################
    resume_metadata = {"resumed_from_existing_state": False}
    run_has_evaluation = os.path.exists(os.path.join(runs_dir, "evaluation_results"))
    run_has_seed_summary = os.path.exists(os.path.join(runs_dir, "seed_summary.json"))
    run_completed = os.path.exists(runs_dir) and run_has_seed_summary and (run_has_evaluation or skip_final_evaluation)
    if os.path.exists(runs_dir) and not run_completed:
        resume_state_available = os.path.exists(os.path.join(runs_dir, "gepa_state.bin")) and os.path.exists(os.path.join(runs_dir, "prog_candidates"))
        can_rebuild_summary = run_has_evaluation or (skip_final_evaluation and resume_state_available)
        if can_rebuild_summary and resume_incomplete:
            print(f"Run directory {runs_dir} already has evaluation results but is missing summary sidecars. Rebuilding summaries from existing artifacts...")
            write_run_summaries(
                runs_dir=runs_dir,
                benchmark=benchmark,
                benchmark_name=benchmark_name,
                prog_name=prog_name,
                optim_name=optim_name,
                lm_name=lm_name,
                lm_config=lm_config,
                judge_lm_config=judge_lm_config,
                selection_mode=selection_mode,
                validation_sidecar_judge_alignment=validation_sidecar_judge_alignment,
                always_validate_for_teacher_memory=always_validate_for_teacher_memory,
                combined_score_mode=combined_score_mode,
                combined_validation_weight=combined_validation_weight,
                combined_judge_weight=combined_judge_weight,
                combined_validation_score_scale=combined_validation_score_scale,
                combined_min_surrogate_gain=combined_min_surrogate_gain,
                retained_validation_fraction=retained_validation_fraction,
                seed=seed,
                setting_name=setting_name,
                smoke_subset_payload=smoke_subset_payload,
                validation_subset_payload=validation_subset_payload,
                phase_wall_clock=load_json_from_path(os.path.join(runs_dir, "phase_wall_clock.json")),
                run_name=run_name,
                resume_metadata={"resumed_from_existing_state": False, "recovered_summary_from_existing_evaluation": True},
                skip_final_evaluation=skip_final_evaluation,
            )
            print(f"Recovered summary sidecars for {runs_dir}.")
            return
        if (not run_has_evaluation or skip_final_evaluation) and resume_incomplete and resume_state_available:
            print(f"Run directory {runs_dir} already exists without final evaluation. Resuming from saved GEPA state...")
            resume_metadata = prepare_resume_sidecars(runs_dir)
            directory_existed = False
        else:
            # Move the existing directory to a backup location with a timestamp
            import shutil
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{runs_dir}_backup_{timestamp}"
            print(f"Run directory {runs_dir} already exists but is incomplete. Moving to {backup_dir}...")
            try:
                os.rename(runs_dir, backup_dir)
                print(f"Fast-moved {runs_dir} to {backup_dir} via rename.")
            except OSError:
                shutil.move(runs_dir, backup_dir)

            directory_existed = False
    else:
        directory_existed = run_completed

    os.makedirs(runs_dir, exist_ok=True)
    if directory_existed:
        print(f"Run directory {runs_dir} already exists. Skipping...")
        return

    write_json_to_path(os.path.join(runs_dir, "benchmark_subset_ids.json"), smoke_subset_payload)
    write_json_to_path(os.path.join(runs_dir, "validation_subset_ids.json"), validation_subset_payload)
    run_manifest = {
        "benchmark_name": benchmark_name,
        "program_name": prog_name,
        "optimizer_name": optim_name,
        "lm_name": lm_name,
        "seed": seed,
        "setting_name": setting_name,
        "selection_mode": selection_mode,
        "judge_memory_top_k": judge_memory_top_k,
        "judge_memory_same_predictor_only": judge_memory_same_predictor_only,
        "judge_learned_guide_path": judge_learned_guide_path,
        "judge_strict_learned_guide": judge_strict_learned_guide,
        "validation_sidecar_judge_alignment": validation_sidecar_judge_alignment,
        "always_validate_for_teacher_memory": always_validate_for_teacher_memory,
        "combined_score_mode": combined_score_mode,
        "combined_validation_weight": combined_validation_weight,
        "combined_judge_weight": combined_judge_weight,
        "combined_validation_score_scale": combined_validation_score_scale,
        "combined_min_surrogate_gain": combined_min_surrogate_gain,
        "retained_validation_fraction": retained_validation_fraction,
        "validation_sampling_mode": validation_sampling_mode,
        "validation_subset_seed": validation_subset_payload.get("selection_seed"),
        "skip_final_evaluation": skip_final_evaluation,
        "benchmark_subset": smoke_subset_payload,
        "validation_subset": validation_subset_payload,
        "dataset_source": getattr(benchmark, "dataset_source", None),
        "split_protocol": getattr(benchmark, "split_protocol", None),
        "final_evaluation_cache_disabled": bool(getattr(benchmark, "disable_cache_for_final_evaluation", False)),
        "lm_io_trace_enabled": log_all_io_effective,
        "lm_io_trace_path": lm_io_trace_path,
        "metric_log_examples_enabled": log_all_io_effective,
        "metric_log_traces_enabled": log_all_io_effective,
        "launch_arbor_requested": launch_arbor_requested,
        "launch_arbor_effective": launch_arbor_effective,
        "launch_arbor_disable_reason": launch_arbor_disable_reason,
        "split_sizes": {
            "train": len(benchmark.train_set),
            "val": len(benchmark.val_set),
            "test": len(benchmark.test_set),
        },
        "cache_dir": cache_dir,
        "use_cache_from_opt": use_cache_from_opt,
        "run_name": run_name,
        "resume_incomplete": resume_incomplete,
        **resume_metadata,
    }
    if override_max_metric_calls is not None:
        run_manifest["override_max_metric_calls"] = int(override_max_metric_calls)
    if override_max_total_api_calls is not None:
        run_manifest["override_max_total_api_calls"] = int(override_max_total_api_calls)
    if override_max_total_search_tokens is not None:
        run_manifest["override_max_total_search_tokens"] = int(override_max_total_search_tokens)
    if api_call_hard_limit is not None:
        run_manifest["api_call_hard_limit"] = int(api_call_hard_limit)
    if override_max_search_iterations is not None:
        run_manifest["override_max_search_iterations"] = int(override_max_search_iterations)
    if judge_lm_config is not None:
        run_manifest["judge_lm_name"] = judge_lm_config.get("name")
        run_manifest["judge_lm_model"] = judge_lm_config.get("model")
    write_json_to_path(os.path.join(runs_dir, "run_manifest.json"), run_manifest)

    print("Running", benchmark_name, prog_name, optim_name, lm_name, evalsetname, "seed", seed)

    try:
        if optimizer_config is not None and "launch_arbor" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["launch_arbor"]:
            from gepa_artifact.utils.arbor_runner import ArborRunner
            if "GRPO" in optim_name:
                arbor_config_file_path = os.path.join(os.getcwd(), "utils/arbor/arbor_train.yaml")
                num_gpus = 3
            else:
                import torch
                num_gpus = torch.cuda.device_count()
                if num_gpus == 4:
                    arbor_config_file_path = os.path.join(os.getcwd(), "utils/arbor/arbor_inference.yaml")
                elif num_gpus == 2:
                    arbor_config_file_path = os.path.join(os.getcwd(), "utils/arbor/arbor_inference_2_gpus.yaml")
                else:
                    raise ValueError(f"Number of GPUs {num_gpus} not supported")

            arbor_config = {"config_filepath": arbor_config_file_path, "gpus": list(range(num_gpus))}

            portnum = get_free_port()
            arbor_config["portnum"] = portnum
            arbor_runner_context = ArborRunner(arbor_config["config_filepath"], arbor_config["portnum"], runs_dir)
            arbor_runner_context.__enter__()

            assert "{portnum}" in lm_config["api_base"]
            lm_config["api_base"] = lm_config["api_base"].format(portnum=arbor_config["portnum"])
            if judge_lm_config is not None and "api_base" in judge_lm_config and "{portnum}" in judge_lm_config["api_base"]:
                judge_lm_config["api_base"] = judge_lm_config["api_base"].format(portnum=arbor_config["portnum"])

        metric_counter = CounterWithLock()

        print(f"Benchmark {benchmark_name} contains {len(benchmark.train_set)} train examples, {len(benchmark.val_set)} val examples and {len(benchmark.test_set)} test examples.")

        if dry_run:
            benchmark.train_set = benchmark.train_set[:2]
            benchmark.val_set = benchmark.val_set[:2]
            benchmark.dev_set = benchmark.dev_set[:2]
            benchmark.test_set = benchmark.test_set[:2]
            print(f"Dry run: only using 2 examples from each set.")
            run_manifest["split_sizes"] = {
                "train": len(benchmark.train_set),
                "val": len(benchmark.val_set),
                "test": len(benchmark.test_set),
            }
            write_json_to_path(os.path.join(runs_dir, "run_manifest.json"), run_manifest)

        final_eval_set = benchmark.test_set
        phase_wall_clock = {"optimizer_seconds": 0.0, "evaluation_seconds": 0.0}
        optimizer_lm_usage = get_lm_usage_stats(None)
        judge_lm_usage = get_lm_usage_stats(None)
        evaluation_lm_usage = get_lm_usage_stats(None)

        with MetricWithLogger(
            metric_fn=benchmark_meta.metric,
            run_dir=runs_dir,
            counter_with_lock=metric_counter,
            train_dataset=benchmark.train_set,
            val_dataset=benchmark.val_set,
            test_dataset=benchmark.test_set,
            log_trace=log_all_io_effective,
            log_example=log_all_io_effective,
            log_prediction=True,
        ) as metric_fn_with_logger, Logger(os.path.join(runs_dir, "run_log.txt")) as logger: #
            # logger = Logger(os.path.join(runs_dir, "run_log.txt"))
            if optimizer_config is not None and "launch_arbor" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["launch_arbor"]:
                logger.log("Arbor in session:", arbor_runner_context.session_name)
            elif launch_arbor_requested and launch_arbor_disable_reason:
                logger.log("Arbor requested but disabled:", launch_arbor_disable_reason)

            #######################
            # For GEPA, if feedback_fn_maps is not provided, we create a default feedback function based on metric_with_feedback
            # and apply it to all predictors in the program.
            #######################
            if run_uses_sidecar_judge(selection_mode, validation_sidecar_judge_alignment) and "GEPA" not in optim_name:
                raise ValueError("judge-based selection or warmup alignment sidecar is only supported for GEPA optimizers")

            if "GEPA" in optim_name:
                gepa_runtime_init_args = {}
                if benchmark_meta.feedback_fn_maps is None or benchmark_meta.feedback_fn_maps[program_idx] is None:
                    def feedback_func(predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace):
                        pred = benchmark_meta.metric_with_feedback(module_inputs, module_outputs, None)
                        return {
                            "feedback_score": pred.score,
                            "feedback_text": pred.feedback,
                        }

                    feedback_fn_map = {k:feedback_func for k, v in program.named_predictors()}
                else:
                    feedback_fn_map = benchmark_meta.feedback_fn_maps[program_idx]
                    assert all(k in feedback_fn_map for k, _ in program.named_predictors())

                gepa_runtime_init_args.update({
                    "named_predictor_to_feedback_fn_map": feedback_fn_map,
                    "knowledgebase_qe": None,
                    "logger": logger,
                    "run_dir": runs_dir,
                    "use_wandb": use_wandb,
                    "wandb_api_key": wandb_api_key,
                    "selection_mode": selection_mode,
                    "judge_memory_top_k": judge_memory_top_k,
                    "judge_memory_same_predictor_only": judge_memory_same_predictor_only,
                    "judge_learned_guide_path": judge_learned_guide_path,
                    "judge_strict_learned_guide": judge_strict_learned_guide,
                    "validation_sidecar_judge_alignment": validation_sidecar_judge_alignment,
                    "always_validate_for_teacher_memory": always_validate_for_teacher_memory,
                    "combined_score_mode": combined_score_mode,
                    "combined_validation_weight": combined_validation_weight,
                    "combined_judge_weight": combined_judge_weight,
                    "combined_validation_score_scale": combined_validation_score_scale,
                    "combined_min_surrogate_gain": combined_min_surrogate_gain,
                    "retained_validation_fraction": retained_validation_fraction,
                    "validation_sampling_mode": validation_sampling_mode,
                })

                logger.log("Optimizer config:", optimizer_config)

            if not os.path.exists(os.path.join(runs_dir, "config.json")):
                with open(os.path.join(runs_dir, "config.json"), "w") as f:
                    json.dump({
                        "benchmark_name": benchmark_name,
                        "program_name": prog_name,
                        "program": program,
                        "optimizer_name": optim_name,
                        "lm_name": lm_name,
                        "lm_config": lm_config,
                        "judge_lm_config": judge_lm_config,
                        "judge_memory_top_k": judge_memory_top_k,
                        "judge_memory_same_predictor_only": judge_memory_same_predictor_only,
                        "judge_learned_guide_path": judge_learned_guide_path,
                        "judge_strict_learned_guide": judge_strict_learned_guide,
                        "validation_sidecar_judge_alignment": validation_sidecar_judge_alignment,
                        "always_validate_for_teacher_memory": always_validate_for_teacher_memory,
                        "combined_score_mode": combined_score_mode,
                        "combined_validation_weight": combined_validation_weight,
                        "combined_judge_weight": combined_judge_weight,
                        "combined_validation_score_scale": combined_validation_score_scale,
                        "combined_min_surrogate_gain": combined_min_surrogate_gain,
                        "num_threads": num_threads,
                        "optimizer_config": optimizer_config,
                        "launch_arbor_requested": launch_arbor_requested,
                        "launch_arbor_effective": launch_arbor_effective,
                        "launch_arbor_disable_reason": launch_arbor_disable_reason,
                        "metric_lm_name": metric_lm_name,
                        "metric_lm": metric_lm,
                        "setting_name": setting_name,
                        "selection_mode": selection_mode,
                        "retained_validation_fraction": retained_validation_fraction,
                        "validation_sampling_mode": validation_sampling_mode,
                        "validation_subset_seed": validation_subset_payload.get("selection_seed"),
                        "skip_final_evaluation": skip_final_evaluation,
                    }, f, default=json_encoder)

            eval_results = EvaluationResult(
                benchmark=benchmark_name,
                program=prog_name,
            )

            if optim_name == "Baseline" or optimizer_config is None:
                # Only run the final evaluation
                optimized_program = program
                eval_results.optimized_program = optimized_program
            else:
                # Run the optimizer, and then run the final evaluation
                optimizer = optimizer_config.optimizer
                init_args = copy.deepcopy(optimizer_config.init_args)
                if "GEPA" in optim_name:
                    init_args.update(gepa_runtime_init_args)
                compile_args = copy.deepcopy(optimizer_config.compile_args)
                langProBe_configs = copy.deepcopy(optimizer_config.langProBe_configs) | {"name": optimizer_config.name}
                judge_lm = None

                #######################
                # Add various configurations to the init_args of the optimizer
                #######################
                if num_threads and "num_threads" in init_args:
                    init_args["num_threads"] = num_threads
                if "provide_logdir_in_init" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["provide_logdir_in_init"]:
                    init_args["log_dir"] = os.path.join(runs_dir, "optimizer_logs")
                    os.makedirs(init_args["log_dir"], exist_ok=True)

                if "add_max_errors_to_initargs" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["add_max_errors_to_initargs"]:
                    init_args["max_errors"] = (len(benchmark.train_set) + len(benchmark.val_set)) * 100

                if "add_max_metric_calls" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["add_max_metric_calls"]:
                    # f"{benchmark_name}_{prog_name}_{optim_name}_{lm_name}"
                    if override_max_total_api_calls is None and override_max_total_search_tokens is None and override_max_search_iterations is None:
                        num_mipro_invocations = get_max_invocations(benchmark_name, prog_name, metric_lm_name, opt=optimizer_config.langProBe_configs.get("max_metric_calls_source_opt_name"))
                        assert num_mipro_invocations is not None, f"Could not find max invocations for {benchmark_name}, {prog_name}, {metric_lm_name}"
                        init_args["max_metric_calls"] = num_mipro_invocations

                if override_max_metric_calls is not None:
                    init_args.pop("max_total_api_calls", None)
                    init_args.pop("max_total_search_tokens", None)
                    init_args.pop("max_search_iterations", None)
                    init_args["max_metric_calls"] = int(override_max_metric_calls)
                if override_max_total_api_calls is not None:
                    init_args.pop("max_metric_calls", None)
                    init_args.pop("max_evals_per_trainval_instance", None)
                    init_args.pop("num_iters", None)
                    init_args.pop("max_total_search_tokens", None)
                    init_args.pop("max_search_iterations", None)
                    init_args["max_total_api_calls"] = int(override_max_total_api_calls)
                if override_max_total_search_tokens is not None:
                    init_args.pop("max_metric_calls", None)
                    init_args.pop("max_evals_per_trainval_instance", None)
                    init_args.pop("num_iters", None)
                    init_args.pop("max_total_api_calls", None)
                    init_args.pop("max_search_iterations", None)
                    init_args["max_total_search_tokens"] = int(override_max_total_search_tokens)
                if api_call_hard_limit is not None:
                    init_args["api_call_hard_limit"] = int(api_call_hard_limit)
                if override_max_search_iterations is not None:
                    init_args.pop("max_metric_calls", None)
                    init_args.pop("max_evals_per_trainval_instance", None)
                    init_args.pop("num_iters", None)
                    init_args.pop("max_total_api_calls", None)
                    init_args.pop("max_total_search_tokens", None)
                    init_args["max_search_iterations"] = int(override_max_search_iterations)

                if "add_wandb_configs_to_initargs" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["add_wandb_configs_to_initargs"]:
                    init_args["use_wandb"] = use_wandb
                    if use_wandb:
                        init_args["wandb_api_key"] = wandb_api_key
                        init_args["wandb_run_name"] = run_name + "_seed_" + str(seed)
                        init_args["wandb_project_name"] = "GEPA"

                if "exclude_seed_from_initargs" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["exclude_seed_from_initargs"]:
                    init_args.pop("seed", None)
                else:
                    init_args['seed'] = seed

                if "GEPA" in optim_name and run_uses_sidecar_judge(selection_mode, validation_sidecar_judge_alignment):
                    judge_lm = create_lm(judge_lm_config or lm_config)
                    init_args["judge_lm"] = judge_lm
                if "GEPA" in optim_name:
                    init_args["validation_sidecar_judge_alignment"] = validation_sidecar_judge_alignment

                optimizer = optimizer(metric=metric_fn_with_logger, **init_args)
                lm_for_optimizer = create_lm(lm_config)
                dspy.configure(lm=lm_for_optimizer, adapter=adapter)

                if "set_lm_before_optimizer" in langProBe_configs and langProBe_configs["set_lm_before_optimizer"]:
                    program.set_lm(lm_for_optimizer)

                print("STARTING COMPILATION FOR", benchmark_name, prog_name, optim_name, lm_name, evalsetname, "seed", seed)
                optimizer_phase_start = time.perf_counter()

                if "add_valset_to_trainset" in langProBe_configs and langProBe_configs["add_valset_to_trainset"]:
                    assert "use_valset" not in langProBe_configs or not langProBe_configs["use_valset"]
                    with lm_trace_context(
                        "optimization",
                        benchmark_name=benchmark_name,
                        program_name=prog_name,
                        optimizer_name=optim_name,
                    ):
                        optimized_program = optimizer.compile(
                            program,
                            trainset = benchmark.train_set + benchmark.val_set,
                            **compile_args,
                        )
                elif "use_valset" in langProBe_configs and langProBe_configs["use_valset"]:
                    with lm_trace_context(
                        "optimization",
                        benchmark_name=benchmark_name,
                        program_name=prog_name,
                        optimizer_name=optim_name,
                    ):
                        optimized_program = optimizer.compile(
                            program,
                            trainset=benchmark.train_set,
                            valset=benchmark.val_set,
                            **compile_args,
                        )
                else:
                    assert False
                phase_wall_clock["optimizer_seconds"] = time.perf_counter() - optimizer_phase_start

                if "use_model_name_from_optimized_program" in langProBe_configs and langProBe_configs["use_model_name_from_optimized_program"]:
                    lm_config["new_model_name"] = lm_for_optimizer.model
                    with open(os.path.join(runs_dir, "lm_config.json"), "w") as f:
                        json.dump(lm_config, f, default=json_encoder)

                optimizer_lm_usage = get_lm_usage_stats(lm_for_optimizer)
                judge_lm_usage = get_lm_usage_stats(judge_lm)
                (
                    eval_results.optimizer_cost,
                    eval_results.optimizer_input_tokens,
                    eval_results.optimizer_output_tokens,
                ) = calculate_stats([lm_for_optimizer, judge_lm])

                eval_results.optimizer = optim_name
                eval_results.optimized_program = optimized_program

                dspy.configure(lm=None, adapter=None)
                del lm_for_optimizer
                if judge_lm is not None:
                    del judge_lm

            summary_eval_results = eval_results
            if skip_final_evaluation:
                summary_eval_results = None
                logger.log("Skipping final test evaluation for this run; writing search-only summaries.")
            else:
                evaluate_prog = dspy.Evaluate(
                    devset=final_eval_set,
                    metric=metric_fn_with_logger,
                    num_threads=num_threads,
                    display_progress=True,
                    max_errors=len(final_eval_set)*10,
                    provide_traceback=True,
                )

                eval_lm_config = copy.deepcopy(lm_config)
                if getattr(benchmark, "disable_cache_for_final_evaluation", False):
                    eval_lm_config["cache"] = False
                    eval_lm_config["cache_in_memory"] = False
                    logger.log(
                        "Disabling LM cache for final evaluation to preserve repeated-test sampling semantics."
                    )

                eval_lm = create_lm(eval_lm_config)
                dspy.configure(lm=eval_lm, adapter=adapter)
                evaluation_phase_start = time.perf_counter()
                with lm_trace_context(
                    "final_test_eval",
                    benchmark_name=benchmark_name,
                    program_name=prog_name,
                    optimizer_name=optim_name,
                ):
                    score_result = evaluate_prog(optimized_program)
                phase_wall_clock["evaluation_seconds"] = time.perf_counter() - evaluation_phase_start
                eval_results.score = getattr(score_result, "score", score_result)
                eval_results.cost, eval_results.input_tokens, eval_results.output_tokens = calculate_stats(
                    eval_lm
                )
                evaluation_lm_usage = get_lm_usage_stats(eval_lm)

                dspy.configure(lm=None, adapter=None)
                del eval_lm

                write_evaluation_result_to_path(
                    eval_results,
                    os.path.join(runs_dir, "evaluation_results"),
                )
            phase_wall_clock["total_seconds"] = phase_wall_clock["optimizer_seconds"] + phase_wall_clock["evaluation_seconds"]
            write_json_to_path(os.path.join(runs_dir, "phase_wall_clock.json"), phase_wall_clock)
            write_run_summaries(
                runs_dir=runs_dir,
                benchmark=benchmark,
                benchmark_name=benchmark_name,
                prog_name=prog_name,
                optim_name=optim_name,
                lm_name=lm_name,
                lm_config=lm_config,
                judge_lm_config=judge_lm_config,
                selection_mode=selection_mode,
                validation_sidecar_judge_alignment=validation_sidecar_judge_alignment,
                always_validate_for_teacher_memory=always_validate_for_teacher_memory,
                combined_score_mode=combined_score_mode,
                combined_validation_weight=combined_validation_weight,
                combined_judge_weight=combined_judge_weight,
                combined_validation_score_scale=combined_validation_score_scale,
                combined_min_surrogate_gain=combined_min_surrogate_gain,
                retained_validation_fraction=retained_validation_fraction,
                seed=seed,
                setting_name=setting_name,
                smoke_subset_payload=smoke_subset_payload,
                validation_subset_payload=validation_subset_payload,
                phase_wall_clock=phase_wall_clock,
                run_name=run_name,
                resume_metadata=resume_metadata,
                skip_final_evaluation=skip_final_evaluation,
                eval_results=summary_eval_results,
                optimizer_lm_usage=optimizer_lm_usage,
                judge_lm_usage=judge_lm_usage,
                evaluation_lm_usage=evaluation_lm_usage,
            )
    finally:
        if optimizer_config is not None and "launch_arbor" in optimizer_config.langProBe_configs and optimizer_config.langProBe_configs["launch_arbor"]:
            arbor_runner_context.__exit__(None, None, None)

def run_experiment_and_write_results(*args, **kwargs):
    try:
        return run_experiment_and_write_results_actual(*args, **kwargs)
    except Exception as e:
        print(traceback.format_exc())
        raise e

def parse_arguments():
    parser = argparse.ArgumentParser(description='A program with boolean arguments.')

    # Argument 1: dry_run
    # Defaults to False. If --dry_run is passed, it becomes True.
    parser.add_argument(
        '--dry_run',
        action='store_true',
        default=False,
        help='Set to true for a dry run (default: False)'
    )

    parser.add_argument('--bm_idx', type=int, required=True, help='Index of the benchmark to run')
    parser.add_argument('--benchmark_name', type=str, required=True, help='Name of the benchmark to run')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use for the benchmark (default: 1)')
    parser.add_argument('--program_idx', type=int, required=True, help='Index of the program to run')
    parser.add_argument('--prog_name', type=str, required=True, help='Name of the program to run')
    parser.add_argument('--opt_idx', type=int, required=True, help='Index of the optimizer to run')
    parser.add_argument('--optim_name', type=str, required=True, help='Name of the optimizer to run')
    parser.add_argument('--lm_config', type=json.loads, required=True, help='JSON string of the LM configuration')
    parser.add_argument('--use_cache_from_opt', type=str, default=None, help='Name of the optimizer to use cache from (default: None)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (default: 0)')
    parser.add_argument('--setting_name', type=str, default=None, help='Optional suffix for run naming and summaries')
    parser.add_argument('--retained_validation_fraction', type=float, default=100.0, help='Retained validation fraction in percent, e.g. 100, 75, 50, 25')
    parser.add_argument('--validation_sampling_mode', type=str, default='fixed',
                        choices=['fixed'],
                        help='Validation examples are randomly selected once at run start and then fixed for the whole run.')
    parser.add_argument('--validation_subset_seed', type=int, default=None,
                        help='Optional seed for the retained validation subset. If omitted, a random seed is generated per new run.')
    parser.add_argument(
        '--selection_mode',
        type=str,
        default=VALIDATION_SELECTION_MODE,
        choices=[
            VALIDATION_SELECTION_MODE,
            FEEDBACK_ONLY_LLM_JUDGE_MODE,
            LEGACY_SCORE_AWARE_LLM_JUDGE_MODE,
            VALIDATION_LLM_JUDGE_COMBINED_MODE,
        ],
        help='Candidate selection mode for GEPA',
    )
    parser.add_argument('--judge_lm_config', type=json.loads, default=None, help='Optional JSON string for the judge LM configuration')
    parser.add_argument('--judge_learned_guide_path', type=str, default=None, help='Optional distilled warmup judge guide file')
    parser.add_argument('--judge_strict_learned_guide', action=argparse.BooleanOptionalAction, default=False, help='Use only the distilled learned guide, without retrieving teacher/alignment memory')
    parser.add_argument('--judge_memory_top_k', type=int, default=3, help='How many historical teacher-memory cases to retrieve per judge decision')
    parser.add_argument(
        '--judge_memory_same_predictor_only',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Prefer retrieving historical teacher-memory cases from the same predictor when available',
    )
    parser.add_argument(
        '--validation_sidecar_judge_alignment',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='During validation warmup, also run a sidecar judge LM that predicts old/new ranking and writes alignment memory records',
    )
    parser.add_argument(
        '--always_validate_for_teacher_memory',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='During validation warmup, validate minibatch-rejected candidates only to write teacher-memory records',
    )
    parser.add_argument(
        '--combined_score_mode',
        type=str,
        default='normalized',
        choices=['normalized', 'direct'],
        help='For validation_llm_judge_combined: combine raw validation deltas directly or normalize by combined_validation_score_scale',
    )
    parser.add_argument('--combined_validation_weight', type=float, default=1.0, help='Weight for the validation component in combined selection')
    parser.add_argument('--combined_judge_weight', type=float, default=1.0, help='Weight for the signed LLM judge confidence in combined selection')
    parser.add_argument('--combined_validation_score_scale', type=float, default=100.0, help='Score range used to normalize validation deltas in combined selection')
    parser.add_argument('--combined_min_surrogate_gain', type=float, default=0.01, help='Minimum positive surrogate gain for accepted combined-selection candidates')
    parser.add_argument('--override_max_metric_calls', type=int, default=None, help='Optional explicit max_metric_calls override for smoke/probe runs')
    parser.add_argument('--override_max_total_api_calls', type=int, default=None, help='Optional optimizer+judge API-call budget for GEPA search')
    parser.add_argument('--override_max_total_search_tokens', type=int, default=None, help='Optional optimizer+judge token budget for GEPA search')
    parser.add_argument('--api_call_hard_limit', type=int, default=None, help='Optional secondary optimizer+judge API-call hard limit')
    parser.add_argument('--override_max_search_iterations', type=int, default=None, help='Optional GEPA search-iteration budget')
    parser.add_argument('--skip_final_evaluation', action=argparse.BooleanOptionalAction, default=False, help='Skip the final test-set evaluation and write search-only summaries')
    parser.add_argument('--resume_incomplete', action=argparse.BooleanOptionalAction, default=True, help='Resume incomplete runs from saved GEPA state when available')
    parser.add_argument('--smoke_train_size', type=int, default=None, help='Optional runner-layer train subset size for smoke only')
    parser.add_argument('--smoke_val_size', type=int, default=None, help='Optional runner-layer validation subset size for smoke only')
    parser.add_argument('--smoke_test_size', type=int, default=None, help='Optional runner-layer test subset size for smoke only')
    parser.add_argument(
        '--log_all_io',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Record all LM request/response JSONL plus metric examples/traces for optimization, validation, judge, and final-test stages. Defaults to GEPA_LOG_ALL_IO.',
    )

    args = parser.parse_args()

    bootstrap_openai_compatible_env()
    args.lm_config = resolve_api_key_env_vars(args.lm_config)
    args.judge_lm_config = resolve_api_key_env_vars(args.judge_lm_config)

    return args

if __name__ == "__main__":
    bootstrap_openai_compatible_env()
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("Please set OPENAI_API_KEY or PAPILLON_API_KEY before running experiments.")
    if env_flag_is_true("GEPA_USE_WANDB", default=True) and "WANDB_API_KEY" not in os.environ:
        raise ValueError("Please set WANDB_API_KEY or disable wandb with GEPA_USE_WANDB=0 before running experiments.")
    args = parse_arguments()
    run_experiment_and_write_results(
        **args.__dict__,
    )
