"""
Launch API-budgeted 100-full and v5 rules experiments.

This study runs three settings for each benchmark:
- 100_full_validation
- 5%+llm_v5_rules_only
- 5%+llm_v5_rules_fewshot

The search budget counts optimizer+judge API calls only. Final test
evaluation remains outside the stopping budget and is recorded separately.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
RUNS_ROOT = PROJECT_ROOT / "experiment_runs_data" / "experiment_runs"
STUDIES_ROOT = PROJECT_ROOT / "experiment_runs_data" / "studies"
DEFAULT_STUDY_NAME = "api_budget_v5_rules_20260615"
POLL_SECONDS = 30
CHILD_PYTHON = os.environ.get(
    "GEPA_EXPERIMENT_PYTHON",
    "/mnt/home/ganfengrui/venvs/gepa-artifact-py311/bin/python",
)

ENDPOINTS = [
    {"slot": "gpu0_port8000", "gpu": 0, "port": 8000},
    {"slot": "gpu1_port8001", "gpu": 1, "port": 8001},
    {"slot": "gpu2_port8002", "gpu": 2, "port": 8002},
    {"slot": "gpu3_port8003", "gpu": 3, "port": 8003},
]

BENCHMARKS = [
    {
        "key": "hotpot",
        "benchmark_name": "HotpotQABench",
        "program_name": "HotpotMultiHop",
        "api_budget": 6000,
    },
    {
        "key": "hover",
        "benchmark_name": "hoverBench",
        "program_name": "HoverMultiHop",
        "api_budget": 7000,
        "extra_env": {
            "GEPA_HOVER_TRAIN_PATH": "/mnt/home/ganfengrui/.cache/gepa/hover/hover_train_release_v1.1.json",
        },
    },
    {
        "key": "ifbench",
        "benchmark_name": "IFBench",
        "program_name": "IFBenchCoT2StageProgram",
        "api_budget": 3500,
        "extra_env": {"PYTHONPATH": str(PROJECT_ROOT / ".vendor")},
    },
    {
        "key": "aime",
        "benchmark_name": "AIMEBench",
        "program_name": "CoT",
        "api_budget": 1000,
        "lm_overrides": {"max_tokens": 2048},
    },
    {
        "key": "papillon",
        "benchmark_name": "Papillon",
        "program_name": "PAPILLON",
        "api_budget": 5000,
        "extra_env": {"GEPA_PAPILLON_AUX_MODEL": "openai/qwen3-8b"},
    },
]

SETTINGS = [
    {
        "key": "baseline_100_full",
        "display_setting": "100_full_validation",
        "kind": "gepa",
        "retained_validation_fraction": 100.0,
        "selection_mode": "validation",
    },
    {
        "key": "v5_rules_only",
        "display_setting": "5%+llm_v5_rules_only",
        "kind": "hybrid",
        "retained_validation_fraction": 5.0,
        "memory_protocol_version": "mem_llm_v5_rules_only",
        "warmup_search_iterations": 50,
    },
    {
        "key": "v5_rules_fewshot",
        "display_setting": "5%+llm_v5_rules_fewshot",
        "kind": "hybrid",
        "retained_validation_fraction": 5.0,
        "memory_protocol_version": "mem_llm_v5_rules_fewshot",
        "warmup_search_iterations": 50,
    },
]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at_utc"] = utc_now()
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def is_pid_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return False
    try:
        parts = stat_path.read_text(encoding="utf-8").split()
        return len(parts) >= 3 and parts[2] != "Z"
    except OSError:
        return False


def build_lm_config(api_base: str, benchmark: dict) -> dict:
    config = {
        "name": "qwen3-8b",
        "model": "openai/qwen3-8b",
        "api_key": "env:OPENAI_API_KEY",
        "api_base": api_base,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 4096,
        "num_retries": 5,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
        },
    }
    config.update(benchmark.get("lm_overrides", {}))
    return config


def build_env(api_base: str, benchmark: dict) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) if not env.get("PYTHONPATH") else f"{SRC_ROOT}:{env['PYTHONPATH']}"
    env["OPENAI_API_KEY"] = "local-qwen"
    env["PAPILLON_API_KEY"] = "local-qwen"
    env["OPENAI_BASE_URL"] = api_base
    env["OPENAI_API_BASE"] = api_base
    env["PAPILLON_API_BASE"] = api_base
    env["GEPA_LOCAL_QWEN3_API_BASE"] = api_base
    env["GEPA_USE_WANDB"] = "0"
    env["GEPA_ENABLE_LAUNCH_ARBOR"] = "0"
    env["GEPA_LOCAL_QWEN3_ENABLE_THINKING"] = "0"
    env["LOCAL_QWEN3_ENABLE_THINKING"] = "0"
    env["GEPA_LOCAL_QWEN3_STRIP_THINK_OUTPUT"] = "0"
    env["LOCAL_QWEN3_STRIP_THINK_OUTPUT"] = "0"
    env["GEPA_LOG_ALL_IO"] = "1"
    env["GEPA_LM_IO_TRACE_ENABLED"] = "1"
    env["GEPA_MAX_CONTEXT_LENGTH"] = "16384"
    env["GEPA_MAX_CONTEXT_LENGTH_TRAINING"] = "16384"
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")
    env["HF_HOME"] = env.get("HF_HOME", "/mnt/home/ganfengrui/.cache/huggingface")
    env["PYTHONUNBUFFERED"] = "1"
    for key, value in benchmark.get("extra_env", {}).items():
        if key == "PYTHONPATH" and env.get("PYTHONPATH"):
            env[key] = f"{value}:{env[key]}"
        else:
            env[key] = value
    return env


def task_setting_name(task: dict) -> str:
    return f"{task['setting_key']}__api{task['api_budget']}__{task['study_name']}"


def run_dir_for(task: dict) -> Path:
    run_name = (
        f"{task['benchmark_name']}_{task['program_name']}_GEPA_qwen3-8b__"
        f"{task['setting_name']}"
    )
    return RUNS_ROOT / f"seed_{task['seed']}" / run_name


def summary_filename(task: dict) -> str:
    return "hybrid_combined_summary.json" if task["kind"] == "hybrid" else "seed_summary.json"


def task_is_complete(task: dict) -> bool:
    return (Path(task["run_dir"]) / summary_filename(task)).exists()


def build_tasks(study_name: str) -> list[dict]:
    tasks = []
    for setting in SETTINGS:
        for benchmark in BENCHMARKS:
            task = {
                "task_id": f"{setting['key']}__{benchmark['key']}",
                "study_name": study_name,
                "benchmark_key": benchmark["key"],
                "benchmark_name": benchmark["benchmark_name"],
                "program_name": benchmark["program_name"],
                "api_budget": benchmark["api_budget"],
                "setting_key": setting["key"],
                "display_setting": setting["display_setting"],
                "kind": setting["kind"],
                "retained_validation_fraction": setting["retained_validation_fraction"],
                "seed": secrets.randbelow(2**31 - 1),
                "validation_subset_seed": secrets.randbelow(2**31 - 1),
                "extra_env": dict(benchmark.get("extra_env", {})),
                "lm_overrides": dict(benchmark.get("lm_overrides", {})),
                "status": "pending",
            }
            task.update({k: v for k, v in setting.items() if k not in {"key", "display_setting", "kind"}})
            task["setting_name"] = task_setting_name(task)
            task["run_dir"] = str(run_dir_for(task))
            tasks.append(task)
    return tasks


def load_or_create_manifest(path: Path, study_name: str, poll_seconds: int) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["poll_seconds"] = poll_seconds
        return manifest

    return {
        "study_name": study_name,
        "created_at_utc": utc_now(),
        "queue_type": "api_budget_v5_rules",
        "budget_note": "Search budget counts optimizer+judge API calls only; final test evaluation is outside budget.",
        "thinking_enabled": False,
        "log_all_io": True,
        "api_budgets": {bm["benchmark_name"]: bm["api_budget"] for bm in BENCHMARKS},
        "settings": [setting["display_setting"] for setting in SETTINGS],
        "endpoints": ENDPOINTS,
        "poll_seconds": poll_seconds,
        "tasks": build_tasks(study_name),
    }


def build_argv(task: dict, api_base: str) -> list[str]:
    benchmark = next(b for b in BENCHMARKS if b["benchmark_name"] == task["benchmark_name"])
    lm_config = json.dumps(build_lm_config(api_base, benchmark), separators=(",", ":"))
    common = [
        "--bm_idx", "0",
        "--benchmark_name", task["benchmark_name"],
        "--num_threads", "1",
        "--program_idx", "0",
        "--prog_name", task["program_name"],
        "--opt_idx", "3",
        "--optim_name", "GEPA",
        "--lm_config", lm_config,
        "--seed", str(task["seed"]),
        "--setting_name", task["setting_name"],
        "--retained_validation_fraction", str(task["retained_validation_fraction"]),
        "--validation_sampling_mode", "fixed",
        "--validation_subset_seed", str(task["validation_subset_seed"]),
        "--resume_incomplete",
    ]

    if task["kind"] == "hybrid":
        return [
            CHILD_PYTHON, "-m", "scripts.run_hybrid_memory_judge",
            *common,
            "--memory_protocol_version", task["memory_protocol_version"],
            "--warmup_search_iterations", str(task["warmup_search_iterations"]),
            "--total_api_calls", str(task["api_budget"]),
        ]

    return [
        CHILD_PYTHON, "-m", "scripts.run_experiments",
        *common,
        "--selection_mode", task["selection_mode"],
        "--override_max_total_api_calls", str(task["api_budget"]),
        "--log_all_io",
    ]


def launch(task: dict, endpoint: dict, log_dir: Path) -> int:
    api_base = f"http://127.0.0.1:{endpoint['port']}/v1"
    benchmark = next(b for b in BENCHMARKS if b["benchmark_name"] == task["benchmark_name"])
    env = build_env(api_base, benchmark)
    argv = build_argv(task, api_base)
    task["assigned_slot"] = endpoint["slot"]
    task["gpu"] = endpoint["gpu"]
    task["port"] = endpoint["port"]
    task["api_base"] = api_base
    task["argv"] = argv
    stdout_path = log_dir / f"{task['task_id']}.stdout.log"
    stderr_path = log_dir / f"{task['task_id']}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            argv,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    task["stdout_log"] = str(stdout_path)
    task["stderr_log"] = str(stderr_path)
    return proc.pid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    study_dir = STUDIES_ROOT / args.study_name
    log_dir = study_dir / "task_logs"
    manifest_path = study_dir / "launch_manifest.json"
    manifest = load_or_create_manifest(manifest_path, args.study_name, args.poll_seconds)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    pending = []
    active: dict[int, dict] = {}
    slot_active: dict[str, int] = {}

    for task in manifest["tasks"]:
        if task_is_complete(task):
            task["status"] = "skipped_completed"
            task.pop("pid", None)
            continue
        if task.get("status") == "running" and task.get("pid") and is_pid_alive(int(task["pid"])):
            pid = int(task["pid"])
            active[pid] = task
            slot_active[task["assigned_slot"]] = pid
            continue
        if task.get("status") != "failed":
            task["status"] = "pending"
            task.pop("pid", None)
            pending.append(task)

    write_json(manifest_path, manifest)

    while pending or active:
        for endpoint in ENDPOINTS:
            if endpoint["slot"] in slot_active or not pending:
                continue
            task = pending.pop(0)
            pid = launch(task, endpoint, log_dir)
            task["pid"] = pid
            task["status"] = "running"
            task["started_at_utc"] = utc_now()
            active[pid] = task
            slot_active[endpoint["slot"]] = pid
            print(f"[{utc_now()}] launched {task['task_id']} pid={pid} {endpoint['slot']}", flush=True)
            write_json(manifest_path, manifest)

        if not active:
            break

        time.sleep(args.poll_seconds)
        for pid, task in list(active.items()):
            if is_pid_alive(pid):
                continue
            slot_active.pop(task.get("assigned_slot"), None)
            if task_is_complete(task):
                task["status"] = "completed"
                task["completed_at_utc"] = utc_now()
            else:
                task["status"] = "failed"
                task["failed_at_utc"] = utc_now()
            task.pop("pid", None)
            active.pop(pid, None)
            print(f"[{utc_now()}] {task['status']} {task['task_id']}", flush=True)
            write_json(manifest_path, manifest)

    write_json(manifest_path, manifest)
    failed = [task["task_id"] for task in manifest["tasks"] if task.get("status") == "failed"]
    print(json.dumps({"study_name": args.study_name, "failed": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
