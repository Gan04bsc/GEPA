# GEPA

Standalone GEPA experiment runner for validation-decay and LLM-judge prompt optimization. The repository contains the runnable GEPA optimizer, benchmark adapters, configuration layer, and launch scripts. Data, caches, logs, and experiment outputs are intentionally kept outside the repository.

## Features

- Validation-only prompt selection with configurable retained validation fraction.
- LLM judge prompt selection with `v1`, `v2`, `v3`, and `combined` strategies.
- Warm-up protocols: use validation decay for N rollouts, distill teacher prompt pairs, then switch to LLM judge.
- API-call, search-iteration, metric-call, and wall-time budget configuration.
- Cost accounting buckets for optimization, judge, minibatch, validation, and final evaluation.
- Benchmark adapters for HotpotQA, HoVer, IFBench, AIME, Papillon, and LiveBench Math.

## Repository Layout

```text
GEPA/
  README.md
  pyproject.toml
  configs/
    experiment.yaml
  scripts/
    run_experiments.py
    run_hybrid_memory_judge.py
    experiment_configs.py
    local_qwen.py
  src/
    gepa_core/        # YAML config, strategy resolution, CLI, backend command builder
    gepa_artifact/    # GEPA optimizer, benchmark adapters, metrics, judge memory
```

The repository does not include benchmark data, retrieval indices, caches, experiment runs, logs, notebooks, or generated figures.

## Install

```bash
git clone <your-repo-url> GEPA
cd GEPA
python3 -m venv ../gepa-venv
source ../gepa-venv/bin/activate
pip install -e .
```

For quick syntax/config checks without installation:

```bash
PYTHONPATH=src python3 -m gepa_core.cli --config configs/experiment.yaml --dry-run
```

## External Assets

Use environment variables to keep data and results outside the repository:

```bash
export GEPA_EXPERIMENT_DIR="$PWD/../gepa-runs"
export HF_HOME="$PWD/../hf-cache"
export GEPA_IFBENCH_DATA_DIR="$PWD/../gepa-data/ifbench"
export GEPA_HOVER_ASSET_DIR="$PWD/../gepa-data/hover"
export HOTPOTQA_QA_DIR="$PWD/../gepa-data/hotpotqa/qa"
```

Benchmark data behavior:

- HotpotQA loads local files from `HOTPOTQA_QA_DIR` if present, otherwise Hugging Face `hotpot_qa/fullwiki`.
- HoVer downloads/uses the training JSON and Wikipedia abstract archive under cache/data locations; set `GEPA_HOVER_TRAIN_PATH` and `GEPA_HOVER_ASSET_DIR` for reproducibility.
- IFBench reads `IFBench_train.jsonl` and `IFBench_test.jsonl` from `GEPA_IFBENCH_DATA_DIR`, or attempts the Hugging Face dataset fallback.
- AIME and Papillon use Hugging Face datasets and respect `HF_HOME`.

## Configure LMs

For OpenAI-compatible local or remote servers:

```bash
export OPENAI_API_KEY="your_key_or_dummy_for_local_server"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export GEPA_LOCAL_QWEN3_API_BASE="http://127.0.0.1:8000/v1"
export GEPA_USE_WANDB=0
export GEPA_ENABLE_LAUNCH_ARBOR=0
```

`GEPA_ENABLE_LAUNCH_ARBOR=0` is recommended unless you have separately configured Arbor/vLLM orchestration.

## YAML Configuration

Main config: `configs/experiment.yaml`.

Important fields:

| Field | Meaning |
| --- | --- |
| `experiment.backend.type` | `dry_run` resolves the plan only; `artifact` launches this repository's runnable scripts. |
| `experiment.backend.artifact_root` | Optional. `null` means this repository root. |
| `experiment.backend.*_index/name` | Benchmark/program/optimizer selector passed to the runnable scripts. |
| `paths.result_dir` | Experiment output directory outside the repo. Also passed as `GEPA_EXPERIMENT_DIR`. |
| `budget.max_llm_calls` | Total optimizer + judge API-call budget. |
| `budget.max_search_iterations` | Search-iteration budget. |
| `budget.max_metric_calls` | Metric-call budget. |
| `validation.retained_fraction` | Validation fraction, e.g. `1.0`, `0.25`, `0.05`, `0.0`. |
| `validation.sampling_mode` | `fixed`. The retained validation subset is sampled once per run and reused throughout the run. |
| `judge.enabled` | Enables LLM judge selection. |
| `judge.version` | `v1`, `v2`, `v3`, or `combined`. |
| `judge.warmup_rollouts` | `0` means pure LLM judge; `>0` means validation warm-up before judge. |
| `judge.combined` | Enables validation + LLM judge combined scoring. |

## Run Modes

### Validate Config Only

```bash
PYTHONPATH=src python3 -m gepa_core.cli --config configs/experiment.yaml --print-plan
PYTHONPATH=src python3 -m gepa_core.cli --config configs/experiment.yaml --dry-run
```

### Pure Validation Decay

Set:

```yaml
experiment:
  backend:
    type: "artifact"
judge:
  enabled: false
validation:
  retained_fraction: 0.05
```

Run:

```bash
gepa-core --config configs/experiment.yaml --dry-run
gepa-core --config configs/experiment.yaml
```

### Pure LLM Judge

Set:

```yaml
judge:
  enabled: true
  version: "v2"
  combined: false
  warmup_rollouts: 0
```

### Warm-up + LLM Judge v3

Set:

```yaml
judge:
  enabled: true
  version: "v3"
  combined: false
  warmup_rollouts: 50
  strict_learned_guide: true
  memory_top_k: 5
budget:
  max_llm_calls: 5000
  max_search_iterations: null
  max_metric_calls: null
```

Semantics:

1. Warm-up uses validation decay.
2. Warm-up records old/new prompt pairs, minibatch feedback, validation scores, and teacher preference.
3. v3 distills a small set of high-signal teacher pairs.
4. Continuation uses only the distilled guide plus current minibatch feedback for LLM judge selection.

### Combined Strategy

Set:

```yaml
judge:
  enabled: true
  version: "combined"
  combined: true
  warmup_rollouts: 0
  combined_strategy:
    validation_weight: 1.0
    judge_weight: 1.0
    normalize_validation_delta: true
```

## Direct Script Entry Points

The CLI is preferred, but scripts can also be run directly:

```bash
PYTHONPATH=src python3 scripts/run_experiments.py --dry_run ...
PYTHONPATH=src python3 scripts/run_hybrid_memory_judge.py --dry_run ...
```

Use `gepa-core --config ... --dry-run` first to print the exact command generated from YAML.

## Extending Strategies

- Add or validate config fields in `src/gepa_core/config.py`.
- Map config to a `StrategyPlan` in `src/gepa_core/strategies.py`.
- Add judge prompt/decision logic in `src/gepa_core/judge.py` or `src/gepa_artifact/gepa/judge_selection.py`.
- Add warm-up memory distillation in `src/gepa_core/memory.py` or `src/gepa_artifact/gepa/judge_memory.py`.
- Add new benchmark adapters under `src/gepa_artifact/benchmarks/`.

## Clean Repository Policy

Do not commit:

- benchmark datasets
- HoVer Wikipedia archives or BM25 indices
- experiment runs
- cache directories
- model outputs
- logs
- notebooks or generated figures
- API keys or `.env` files
