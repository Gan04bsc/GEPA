# GEPA 仓库代码讲解

这份文档面向代码基础较弱的读者，目标是帮你看懂这个仓库每个目录和文件的作用，以及一个 GEPA 实验从配置到运行的大致流程。

你可以先记住一句话：这个仓库不是在从零训练一个大模型，而是在已有 LLM 的基础上，自动搜索和优化 compound AI system 里各个模块的 prompt。

## 1. 先理解整体在做什么

传统深度学习训练通常是：

```text
数据 -> 模型前向传播 -> 计算 loss -> 反向传播 -> 更新模型参数
```

这个仓库做的是 prompt optimization，更像是：

```text
数据 -> AI 系统运行 -> 得到输出 -> 评价函数打分/反馈 -> GEPA 修改 prompt -> 选择更好的 prompt
```

这里的“模型参数”通常不更新。我们主要更新的是 prompt，也就是给 LLM 的指令。

一个实验通常包含这些部分：

```text
配置文件 configs/experiment.yaml
        |
        v
CLI 入口 gepa_core.cli
        |
        v
策略解析 gepa_core.strategies
        |
        v
运行后端 gepa_core.artifact_backend
        |
        v
scripts/run_experiments.py 或 scripts/run_hybrid_memory_judge.py
        |
        v
benchmark 数据集 + compound AI system + metric
        |
        v
GEPA 优化器 gepa_artifact/gepa/gepa.py
        |
        v
结果写入外部 runs 目录
```

## 2. 仓库根目录

```text
GEPA/
  README.md
  CODE_WALKTHROUGH_ZH.md
  pyproject.toml
  .gitignore
  configs/
  scripts/
  src/
```

### README.md

这是给使用者看的快速说明文档，回答这些问题：

- 这个项目是什么。
- 如何安装。
- 如何配置数据路径和模型 API。
- 如何运行纯 validation 实验、纯 LLM judge 实验、warmup + LLM judge 实验。
- 数据、缓存、实验结果应该放在哪里。

如果你只是想知道“怎么跑”，优先看 `README.md`。

### CODE_WALKTHROUGH_ZH.md

就是你现在正在看的这份文档。它的作用是解释代码结构，帮助你理解每个文件在做什么。

### pyproject.toml

这是 Python 项目的安装配置文件。它告诉 Python：

- 这个包叫什么名字。
- 需要哪些依赖。
- 命令行入口是什么。

里面最关键的是：

```toml
[project.scripts]
gepa-core = "gepa_core.cli:main"
```

意思是：安装后你可以在终端运行 `gepa-core`，它会调用 `src/gepa_core/cli.py` 里的 `main()` 函数。

### .gitignore

这个文件告诉 Git 哪些东西不要提交到 GitHub。

例如：

- `data/`
- `runs/`
- `cache/`
- `logs/`
- `__pycache__/`
- `.env`

原因是这些通常是数据、缓存、实验结果或密钥，不应该放进干净代码仓库。

## 3. configs 目录

```text
configs/
  experiment.yaml
```

### configs/experiment.yaml

这是统一实验配置文件。你可以把它理解为“实验控制面板”。

它不写 Python 代码，而是用 YAML 写实验参数。

主要字段如下：

```yaml
experiment:
  name: "example_validation_decay"
  seed: 130
  benchmark: "HotpotQABench"
```

这部分定义实验名称、随机种子和数据集。

```yaml
experiment:
  backend:
    type: "dry_run"
```

`backend.type` 决定是不是真跑实验：

- `dry_run`：只检查配置和打印计划，不真正跑。
- `artifact`：调用真实脚本运行实验。

```yaml
budget:
  max_llm_calls: null
  max_search_iterations: 200
  max_metric_calls: null
```

这部分定义预算：

- `max_llm_calls`：最多允许多少次 LLM API 调用。
- `max_search_iterations`：最多搜索多少轮 prompt。
- `max_metric_calls`：最多评价多少个样本。

```yaml
validation:
  retained_fraction: 0.05
  sampling_mode: "fixed"
```

这部分控制验证集使用比例。`0.05` 就是只用 5% 验证集。

现在的逻辑是 fixed：一次实验开始时随机选一份验证子集，后续一直用这一份，不会每轮重新抽。

```yaml
judge:
  enabled: false
  version: "v1"
  combined: false
  warmup_rollouts: 0
```

这部分控制是否使用 LLM judge：

- `enabled: false`：只用 validation 选 prompt。
- `enabled: true`：启用 LLM judge。
- `warmup_rollouts: 50`：先用 validation warmup 50 轮，再切到 LLM judge。
- `version: "v3"`：使用 v3 这类带 distilled guide 的策略。

```yaml
program:
  optimizer_lm:
    name: "qwen3-8b"
  task_lm:
    name: "qwen3-8b"
```

这部分定义使用哪个 LLM。这里的 LLM 可以同时承担两种角色：

- 作为 AI 系统内部模块的 LLM。
- 作为 GEPA 生成/优化 prompt 的 LLM。

## 4. scripts 目录

```text
scripts/
  experiment_configs.py
  local_qwen.py
  run_experiments.py
  run_hybrid_memory_judge.py
```

`scripts` 目录是“真实实验脚本层”。它比 `gepa_core` 更贴近具体实验实现。

### scripts/experiment_configs.py

这个文件定义可用的数据集、模型和优化器。

你可以理解为“实验注册表”。

它里面有几类重要内容：

```python
LM_CONFIGS = [...]
```

这里注册可用 LLM，例如 `qwen3-8b`、`gpt-4.1-mini`。

```python
def get_benchmarks(...)
```

这个函数返回可用 benchmark，例如：

- HotpotQA
- HoVer
- IFBench
- AIME
- Papillon

```python
def get_optimizers()
```

这个函数返回可用优化器，例如：

- Baseline
- MIPROv2
- GEPA
- GRPO

对初学者来说，这个文件的作用就是：告诉实验脚本“有哪些模型、数据集、优化方法可以选择”。

### scripts/local_qwen.py

这个文件处理本地或远端 Qwen 模型服务的 API 地址。

例如你用 vLLM 部署了一个 OpenAI-compatible server：

```text
http://127.0.0.1:8000/v1
```

这个文件就负责把模型名映射到正确的 API base、extra body 等配置。

### scripts/run_experiments.py

这是单阶段实验的主运行脚本。

所谓单阶段，就是实验从头到尾使用一种选择方式，比如：

- 纯 validation。
- 纯 LLM judge。
- combined validation + LLM judge。

它做的事情大概是：

```text
读取命令行参数
加载 benchmark
加载 program
加载 optimizer
设置 LM
运行优化器
记录 token / API calls / 分数 / prompt
保存结果
```

里面有几个重要概念：

```python
selection_mode = "validation"
```

表示用验证集分数来选择 prompt。

```python
selection_mode = "llm_judge"
```

表示不用完整 validation 判断 prompt，而是让 LLM judge 根据 feedback 判断 old prompt 和 new prompt 谁更好。

```python
get_lm_usage_stats(...)
```

这个函数统计 LLM 的调用成本，例如：

- API 调用次数。
- input tokens。
- output tokens。
- cost。

这是后面分析实验开销的基础。

### scripts/run_hybrid_memory_judge.py

这是两阶段实验脚本，主要用于 v3 这类 warmup + LLM judge 实验。

两阶段意思是：

```text
阶段 1：validation warmup
阶段 2：LLM judge continuation
```

阶段 1 会用 validation 产生一批 prompt pair 和真实验证结果。它会记录：

- old prompt。
- new prompt。
- minibatch feedback。
- old validation score。
- new validation score。
- validation teacher 更喜欢 old 还是 new。

阶段 2 会把 warmup 中的高价值 pair 蒸馏成一个 guide，然后让 LLM judge 参考这个 guide 来判断后续 prompt。

这个文件里比较关键的函数：

```python
copy_required_artifacts(...)
```

把 warmup 阶段产生的重要文件复制到 continuation 阶段。

```python
collapse_validation_frontier_to_surrogate(...)
```

把 warmup 阶段的 validation frontier 转成 continuation 阶段可继续使用的状态。

```python
write_learned_judge_guide_artifacts(...)
```

从 warmup memory 中挑选高价值 pair，生成 `judge_prompt_lessons.md/json`。

```python
build_combined_summary(...)
```

把 warmup 和 continuation 两阶段的结果合并成一个总 summary。

## 5. src/gepa_core 目录

```text
src/gepa_core/
  __init__.py
  accounting.py
  adapters.py
  artifact_backend.py
  budgets.py
  cli.py
  config.py
  judge.py
  memory.py
  runner.py
  strategies.py
  validation.py
```

`gepa_core` 是这个仓库整理出来的“干净核心入口层”。它尽量不写复杂 benchmark 细节，而是负责：

- 读配置。
- 判断实验策略。
- 构造运行命令。
- 提供统一 CLI。
- 抽象成本统计、预算、judge、memory 等概念。

### src/gepa_core/cli.py

这是命令行入口。

当你运行：

```bash
gepa-core --config configs/experiment.yaml
```

实际会进入这里的 `main()`。

它做三件事：

```text
读取 YAML 配置
创建 ExperimentRunner
运行或 dry-run
```

如果加：

```bash
--print-plan
```

它只打印解析后的策略，不启动实验。

### src/gepa_core/config.py

这个文件定义配置结构。

它把 YAML 里的字段转换成 Python dataclass。

例如：

```python
@dataclass(frozen=True)
class BudgetConfig:
    max_llm_calls: int | None = None
    max_search_iterations: int | None = None
    max_metric_calls: int | None = None
```

这表示预算配置有三种常见限制。

再比如：

```python
@dataclass(frozen=True)
class JudgeConfig:
    enabled: bool = False
    version: JudgeVersion = "v1"
    warmup_rollouts: int = 0
```

这表示 LLM judge 是否启用、用哪个版本、warmup 多少轮。

初学者可以这样理解：`config.py` 是把“文本配置文件”变成“Python 能安全使用的对象”。

### src/gepa_core/strategies.py

这个文件根据配置判断实验模式。

它会把配置转成几种模式：

```python
VALIDATION_DECAY
PURE_LLM_JUDGE
WARMUP_THEN_LLM_JUDGE
COMBINED
```

例如：

- `judge.enabled = false` -> 纯 validation。
- `judge.enabled = true` 且 `warmup_rollouts = 0` -> 纯 LLM judge。
- `judge.enabled = true` 且 `warmup_rollouts > 0` -> warmup + LLM judge。
- `judge.combined = true` -> combined 策略。

这层的作用是：让后面的运行器不用自己猜实验类型。

### src/gepa_core/runner.py

这是一个很薄的运行协调器。

它做的事情是：

```text
根据 config 生成 strategy plan
初始化预算检查器
选择 backend
调用 backend.run()
```

你可以把它理解成“实验总控入口”。

### src/gepa_core/adapters.py

这个文件定义后端接口和 dry-run 后端。

里面有：

```python
class ExperimentBackend(Protocol)
```

它规定了所有 backend 都应该有一个 `run()` 方法。

```python
class DryRunBackend
```

它不真正跑实验，只返回“配置解析成功”的结果。

这就是为什么你可以先用：

```bash
gepa-core --config configs/experiment.yaml --dry-run
```

检查配置是否正确。

### src/gepa_core/artifact_backend.py

这个文件负责把干净 YAML 配置翻译成真实脚本命令。

例如它会生成类似：

```bash
python scripts/run_experiments.py \
  --benchmark_name HotpotQABench \
  --selection_mode validation \
  --override_max_search_iterations 200
```

如果是 warmup + LLM judge，它会改成调用：

```bash
scripts/run_hybrid_memory_judge.py
```

这个文件是 `gepa_core` 和 `scripts` 之间的桥梁。

### src/gepa_core/budgets.py

这个文件定义预算检查逻辑。

预算可以包括：

- LLM API 调用次数。
- search iterations。
- metric calls。
- 运行时间。

核心概念是：

```python
BudgetGuard.should_stop(...)
```

它判断实验是否已经达到预算，需要停止。

### src/gepa_core/accounting.py

这个文件定义成本统计结构。

它把成本分成几个 bucket：

```text
optimization
judge
minibatch
validation
evaluation
```

每个 bucket 记录：

- input tokens。
- output tokens。
- API calls。
- seconds。

这些字段对应你前面一直关心的“开销到底花在哪个环节”。

### src/gepa_core/judge.py

这个文件定义 LLM judge 的基础逻辑。

它包括：

```python
build_selection_prompt(...)
```

构造给 judge LLM 的 prompt，让它比较 old prompt 和 new prompt。

```python
parse_judge_decision(...)
```

解析 LLM 返回的 JSON。

```python
combined_delta(...)
```

把 validation delta 和 judge confidence 合成一个 combined 分数。

简单说：这个文件回答“如果让 LLM 当裁判，应该问它什么、怎么解析它的答案、怎么把答案转成选择信号”。

### src/gepa_core/memory.py

这个文件定义 teacher memory 的简化抽象。

teacher memory 就是 warmup 阶段留下来的 prompt pair：

```text
old prompt
new prompt
feedback
old validation score
new validation score
```

它能做几件事：

```python
load_teacher_pairs(...)
```

从 jsonl 文件读取 warmup pair。

```python
select_distilled_pairs(...)
```

按 validation 分差和相似度筛选高价值 pair。

```python
build_learned_guide(...)
```

把筛选出的 pair 写成 LLM judge 可读的 guide。

### src/gepa_core/validation.py

这个文件处理验证集子集选择和 validation 接受逻辑。

```python
select_validation_subset(...)
```

从完整验证集中选一部分样本。

当前只支持 fixed：实验开始时选一次，后续一直用这同一批。

```python
validation_accepts_update(...)
```

判断 new score 是否比 old score 好。

### src/gepa_core/__init__.py

Python 包标记文件。它让 `gepa_core` 可以被当成一个 Python package 导入。

通常里面不需要复杂逻辑。

## 6. src/gepa_artifact 目录

```text
src/gepa_artifact/
  __init__.py
  benchmarks/
  gepa/
  utils/
```

`gepa_artifact` 是更接近原始实验实现的代码层。这里包含真正的 benchmark、GEPA 优化器和工具函数。

如果说 `gepa_core` 是“干净控制层”，那么 `gepa_artifact` 就是“实验执行层”。

## 7. src/gepa_artifact/benchmarks 目录

```text
benchmarks/
  benchmark.py
  dspy_program.py
  AIME/
  IFBench/
  hotpotQA/
  hover/
  livebench_math/
  papillon/
```

这个目录定义不同数据集和 compound AI system。

### benchmarks/benchmark.py

这是 benchmark 的基础抽象。

里面有：

```python
class Benchmark
```

它规定一个数据集应该有：

- `train_set`
- `val_set`
- `test_set`

还规定每个 benchmark 要实现：

```python
init_dataset()
```

也就是如何加载自己的数据。

还有：

```python
BenchmarkMeta
```

它把一个 benchmark 需要的东西打包：

- benchmark 类。
- program。
- metric。
- feedback function。
- 线程数。
- 名称。

你可以把 `BenchmarkMeta` 理解成“一个实验任务的身份证”。

### benchmarks/dspy_program.py

这个文件通常放 DSPy program 相关的通用辅助逻辑。

DSPy 是一个用 Python 组织 LLM pipeline 的框架。这里的 compound AI system 往往是 DSPy module。

### benchmarks/AIME

```text
AIME/
  AIME_data.py
  AIME_program.py
```

AIME 是数学推理任务。

`AIME_data.py` 负责加载数学题数据。

`AIME_program.py` 定义解题系统，一般流程是：

```text
math problem -> LLM reasoning -> final answer
```

评价通常看最终答案是否和 gold answer 一致。

### benchmarks/IFBench

```text
IFBench/
  ifbench_data.py
  ifbench_metric.py
  ifbench_program.py
  utils_ifbench/
```

IFBench 是 instruction following benchmark。

任务是检查模型输出是否满足复杂指令约束，例如：

- 必须包含某些关键词。
- 关键词出现次数必须正确。
- 格式必须符合要求。

`ifbench_data.py` 加载数据。

`ifbench_program.py` 定义回答/重写的 AI 系统。

`ifbench_metric.py` 定义如何判断输出是否满足约束。

`utils_ifbench/` 里是大量具体 instruction checker，例如检查长度、关键词、格式等。

### benchmarks/hotpotQA

```text
hotpotQA/
  hotpot_data.py
  hotpot_program.py
  hotpot_utils.py
```

HotpotQA 是多跳问答任务。

一个典型系统流程是：

```text
question
-> query generation
-> retrieval
-> document summary
-> second-hop query
-> retrieval
-> summary
-> final answer
```

`hotpot_data.py` 负责加载 HotpotQA 数据。

`hotpot_program.py` 定义多模块问答系统。

`hotpot_utils.py` 放检索、答案处理、工具函数等。

### benchmarks/hover

```text
hover/
  hover_data.py
  hover_program.py
  hover_utils.py
```

HoVer 是多跳事实验证任务。

输入是一个 claim，输出通常是：

```text
SUPPORTED 或 NOT_SUPPORTED
```

系统流程类似：

```text
claim
-> query generation
-> retrieval
-> evidence summary
-> reasoning
-> support judgment
```

### benchmarks/papillon

```text
papillon/
  papillon_data.py
  papillon_program.py
  papillon_utils.py
```

Papillon 是隐私保护相关任务。

系统流程通常是：

```text
private user query
-> redaction / rewrite
-> external LM call
-> final answer rewrite
```

评价不只看回答质量，也会关心 privacy / leakage。

### benchmarks/livebench_math

```text
livebench_math/
  livebenchmath_data.py
  livebenchmath_program.py
  livebenchmath_utils/
```

这是另一个数学任务集合。

它和 AIME 类似，重点是数学推理和答案验证。

## 8. src/gepa_artifact/gepa 目录

```text
gepa/
  entropy_utils.py
  gepa.py
  gepa_utils.py
  instruction_proposal.py
  judge_memory.py
  judge_selection.py
  merge_programs.py
```

这是 GEPA 优化器的核心实现。

### gepa/gepa.py

这是最核心的文件。

它实现 GEPA 的主循环，大致逻辑是：

```text
选择一个已有 candidate prompt
选择一个模块 predictor
在 minibatch 上运行，拿到 feedback
调用 instruction proposer 生成 new prompt
评估 new prompt
决定是否接受
保存 candidate、trace、成本和分数
进入下一轮
```

你可以把它理解成 prompt 版本的“训练循环”。

传统深度学习训练循环更新的是模型参数；GEPA 训练循环更新的是 prompt。

### gepa/gepa_utils.py

这个文件放 GEPA 的辅助函数和状态处理。

例如：

- 保存和加载 GEPA state。
- 管理 candidate program。
- 处理 Pareto frontier。
- 辅助统计。

### gepa/instruction_proposal.py

这个文件负责“生成新 prompt”。

GEPA 会根据当前 prompt、错误反馈、样本表现，让 LLM 提出一个修改后的 prompt。

这个过程就像：

```text
旧 prompt + 失败案例 + 反馈 -> LLM 反思 -> 新 prompt
```

### gepa/judge_selection.py

这个文件负责 LLM judge 的 prompt 和解析逻辑。

它会问 judge LLM：

```text
old prompt 和 new prompt 哪个更可能泛化更好？
你的置信度是多少？
原因是什么？
风险是什么？
```

然后解析 LLM 输出的 JSON。

### gepa/judge_memory.py

这个文件负责 memory bank 和 distilled guide。

它处理：

- teacher memory。
- alignment memory。
- 从 warmup 记录中选择高价值 pair。
- 构造 `judge_prompt_lessons.md/json`。

其中 v3/v4 这类策略会用到这里的逻辑。

简单理解：

```text
warmup 产生很多 old/new pair
judge_memory.py 从里面挑少数代表性强的 pair
写成后续 LLM judge 可以学习的 guide
```

### gepa/merge_programs.py

这个文件和 prompt/program 合并有关。

如果实验使用 merge 策略，它会尝试把不同 candidate program 的优点合并。

当前主线如果只跑普通 GEPA 或 v3/v4，不一定重点看这个文件。

### gepa/entropy_utils.py

这个文件放和 entropy 或不确定性相关的工具函数。

这类工具通常用于分析模型输出分布或候选选择的不确定性。

## 9. src/gepa_artifact/utils 目录

```text
utils/
  arbor_runner.py
  capture_stream_logger.py
  json_default_encoder.py
  metric_logger.py
  optimizers.py
```

这是工具函数目录。

### utils/arbor_runner.py

和 Arbor / vLLM 这类模型服务启动或管理相关。

如果你已经手动启动了 OpenAI-compatible API server，通常不需要重点看它。

### utils/capture_stream_logger.py

日志工具。

它可以把 stdout/stderr 输出捕获并写入日志文件，方便实验后排查。

### utils/json_default_encoder.py

JSON 序列化工具。

Python 里有些对象默认不能直接 `json.dump`，这个文件提供自定义编码逻辑。

### utils/metric_logger.py

评价指标日志工具。

它帮助记录每次 metric 调用、分数、检查点等。

### utils/optimizers.py

优化器配置工具。

它定义如何包装不同 optimizer 的初始化参数和运行参数。

## 10. 重要运行文件之间的关系

如果你运行：

```bash
gepa-core --config configs/experiment.yaml --dry-run
```

调用链是：

```text
gepa_core/cli.py
-> gepa_core/config.py
-> gepa_core/runner.py
-> gepa_core/strategies.py
-> gepa_core/adapters.py
```

如果你把 backend 改成 `artifact` 真跑实验：

```text
gepa_core/cli.py
-> gepa_core/artifact_backend.py
-> scripts/run_experiments.py
-> scripts/experiment_configs.py
-> gepa_artifact/benchmarks/*
-> gepa_artifact/gepa/gepa.py
```

如果是 warmup + LLM judge：

```text
gepa_core/artifact_backend.py
-> scripts/run_hybrid_memory_judge.py
-> 先跑 validation warmup
-> 写 judge_memory_bank.jsonl
-> 生成 judge_prompt_lessons.md/json
-> 再跑 LLM judge continuation
```

## 11. 初学者需要理解的几个关键词

### LLM

Large Language Model，大语言模型，例如 Qwen、GPT。

在这个仓库里，LLM 既可以是任务系统内部的模型，也可以是 prompt optimizer / judge。

### Prompt

给 LLM 的自然语言指令。

这个仓库主要优化 prompt，而不是训练模型参数。

### Compound AI System

由多个模块组成的 AI 系统。

例如 HotpotQA 不是只问一次 LLM，而是可能包含：

```text
query writer
retriever
summarizer
answer generator
```

每个模块都可能有自己的 prompt。

### Benchmark

数据集 + 任务定义 + 评价函数。

例如 AIME、HotpotQA、Papillon 都是 benchmark。

### Metric

评价函数。

它把模型输出和标准答案比较，得到分数。

例如：

- AIME：最终数字答案是否正确。
- IFBench：是否满足所有指令约束。
- HotpotQA：答案是否匹配 gold answer。

### Feedback

比单纯分数更详细的错误信息。

例如：

```text
答案格式错了
缺少关键词
没有找到 gold document
最终答案和标准答案不一致
```

GEPA 依赖 feedback 来指导 prompt 修改。

### Validation

验证集。

用于在优化过程中判断 prompt 是否变好。

### Test

测试集。

通常只在最后评估，不应该用于优化 prompt。

### Warmup

先用 validation 跑若干轮，收集 prompt pair 和真实分数。

### Teacher Memory

warmup 期间由 validation 产生的 old/new prompt 对比记录。

因为 validation 分数更接近真实评价，所以叫 teacher。

### LLM Judge

让另一个 LLM 判断 old prompt 和 new prompt 谁更好。

好处是可能省 validation 成本。

风险是 judge 可能判断错。

### Alignment Memory

记录 LLM judge 判断错的案例。

例如 LLM judge 觉得 new prompt 更好，但 validation 发现 old prompt 更好。

这些错误案例可以帮助后续 judge 避免同类错误。

### API Calls

调用 LLM 服务的次数。

一次 prompt 输入给模型并拿到输出，通常算一次 API call。

### Tokens

LLM 的文本计量单位。

输入越长、输出越长，tokens 越多，成本通常越高。

## 12. 怎么从零开始读这个仓库

推荐阅读顺序：

1. 先看 `README.md`，知道如何安装和运行。
2. 看 `configs/experiment.yaml`，理解实验参数。
3. 看 `src/gepa_core/cli.py`，理解命令行入口。
4. 看 `src/gepa_core/config.py`，理解配置怎么变成 Python 对象。
5. 看 `src/gepa_core/strategies.py`，理解不同实验模式怎么区分。
6. 看 `src/gepa_core/artifact_backend.py`，理解 YAML 怎么变成真实命令。
7. 看 `scripts/run_experiments.py`，理解单阶段实验怎么跑。
8. 看 `scripts/run_hybrid_memory_judge.py`，理解 warmup + judge 怎么跑。
9. 看 `src/gepa_artifact/benchmarks/benchmark.py`，理解 benchmark 抽象。
10. 看某一个具体数据集目录，例如 `AIME/` 或 `hotpotQA/`。
11. 最后看 `src/gepa_artifact/gepa/gepa.py`，理解 GEPA 主循环。

不要一开始就读 `gepa.py`，它比较长，直接读会很容易迷路。

## 13. 一个最小运行例子

先只检查配置：

```bash
cd /mnt/home/ganfengrui/GEPA
PYTHONPATH=src python3 -m gepa_core.cli --config configs/experiment.yaml --dry-run
```

如果输出类似：

```json
{
  "status": "dry_run",
  "message": "Resolved mode=validation_decay..."
}
```

说明配置可以被解析。

如果要真跑，需要：

1. 启动 LLM API server。
2. 设置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。
3. 准备 benchmark 数据路径。
4. 把 `configs/experiment.yaml` 里的 backend 改成 `artifact`。

例如：

```yaml
experiment:
  backend:
    type: "artifact"
```

然后运行：

```bash
gepa-core --config configs/experiment.yaml
```

## 14. 如果你想改实验配置

### 改数据集

改：

```yaml
experiment:
  benchmark: "AIMEBench"
```

同时需要对应改 backend 里的 index 和 program name。

### 改验证集比例

改：

```yaml
validation:
  retained_fraction: 0.05
```

常见值：

- `1.0`：100% validation。
- `0.75`：75% validation。
- `0.5`：50% validation。
- `0.25`：25% validation。
- `0.05`：5% validation。

### 改成纯 LLM judge

改：

```yaml
judge:
  enabled: true
  version: "v2"
  combined: false
  warmup_rollouts: 0
```

### 改成 warmup + LLM judge

改：

```yaml
judge:
  enabled: true
  version: "v3"
  combined: false
  warmup_rollouts: 50
  strict_learned_guide: true
```

### 改预算为 API 调用次数

改：

```yaml
budget:
  max_llm_calls: 5000
  max_search_iterations: null
  max_metric_calls: null
```

## 15. 如何理解这个仓库里的“深度学习框架”

这个仓库使用的不是 PyTorch 手写训练循环作为主线，而是 DSPy + LLM API + GEPA。

你可以这样类比：

```text
PyTorch 里的 Module      -> DSPy 里的 Module / Program
训练数据 batch           -> GEPA minibatch
loss                     -> metric / feedback
optimizer.step()         -> GEPA 修改 prompt
模型 checkpoint          -> prompt candidate / optimized program
validation score         -> validation evaluator score
```

最大的区别是：

```text
传统训练：更新神经网络权重
GEPA：更新自然语言 prompt
```

所以代码里大量出现的是：

- prompt。
- predictor。
- program。
- feedback。
- validation score。
- judge decision。
- memory bank。

而不是：

- tensor。
- gradient。
- loss.backward。
- optimizer.step。

## 16. 你应该重点关注哪些输出文件

真实实验结果不会放在这个核心仓库里，而是放在外部 runs 目录。

常见重要文件：

```text
seed_summary.json
```

实验总体结果，包括最终分数、成本、迭代数等。

```text
iteration_summary.jsonl
```

每一轮 GEPA 的摘要。

```text
metric_call_checkpoints.jsonl
```

metric 调用记录，适合分析 validation/minibatch/final eval 的成本。

```text
instruction_proposer_inpouts.jsonl
```

prompt proposer 的输入输出。

```text
judge_decisions.jsonl
```

LLM judge 的判断记录。

```text
judge_memory_bank.jsonl
```

teacher memory 记录。

```text
judge_alignment_memory_bank.jsonl
```

judge 判断错的 alignment memory 记录。

```text
prog_candidates/
```

保存不同版本的 candidate program/prompt。

## 17. 总结

这个仓库可以分成三层：

```text
configs + gepa_core
```

负责干净配置、CLI、策略选择和命令生成。

```text
scripts
```

负责把配置真正落地成实验运行。

```text
gepa_artifact
```

负责 benchmark、GEPA 优化器、LLM judge、memory 和工具函数。

理解这个仓库的关键不是先学会所有代码细节，而是先抓住主线：

```text
配置实验 -> 选择策略 -> 运行 benchmark -> GEPA 生成 prompt -> validation 或 LLM judge 选择 prompt -> 记录成本和分数
```

