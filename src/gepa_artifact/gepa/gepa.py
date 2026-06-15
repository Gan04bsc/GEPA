
from ast import Set
import math
import os
import random
import time
import traceback
from typing import Literal, Union
import dspy
import json

import dspy.teleprompt
import dspy.teleprompt.teleprompt
import wandb

from .instruction_proposal import ProposeNewInstructionModule
from .judge_selection import judge_prompt_candidate, predict_alignment_pairwise_preference
from .judge_memory import (
    delta_bucket,
    format_alignment_memory_records,
    format_memory_records,
    load_memory_bank,
    retrieve_relevant_memory_records,
    summarize_feedback_examples,
    summarize_alignment_memory_bank,
    summarize_memory_bank,
    teacher_label_from_scores,
    teacher_preferred_prompt,
)
from gepa_artifact.utils.lm_io_trace import lm_trace_context
from dspy import Example
from typing import List, Set
from collections import Counter

from .gepa_utils import (
    GEPAState,
    idxmax,
    capture_module_trace_with_feedback,
    select_program_candidate_from_pareto_front,
    log_detailed_metrics_after_discovering_new_program,
    initialize_gepa_state,
    initialize_wandb,
    find_dominator_programs,
    make_selection_surrogate_eval_output,
)

from .merge_programs import (
    sample_and_attempt_merge_programs_by_common_predictors
)

class LegacyEvaluate:
    def __init__(
        self,
        *,
        devset,
        metric=None,
        num_threads=None,
        display_progress=False,
        display_table=False,
        max_errors=None,
        provide_traceback=None,
        failure_score=0.0,
        save_as_csv=None,
        save_as_json=None,
        **kwargs,
    ):
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback
        self.failure_score = failure_score
        self.save_as_csv = save_as_csv
        self.save_as_json = save_as_json
        self.extra_kwargs = dict(kwargs)

    def __call__(self, program):
        evaluation_result = dspy.Evaluate(
            devset=self.devset,
            metric=self.metric,
            num_threads=self.num_threads,
            display_progress=self.display_progress,
            display_table=self.display_table,
            max_errors=self.max_errors,
            provide_traceback=self.provide_traceback,
            failure_score=self.failure_score,
            save_as_csv=self.save_as_csv,
            save_as_json=self.save_as_json,
            **self.extra_kwargs,
        )(program)
        if isinstance(evaluation_result, tuple):
            return evaluation_result
        if isinstance(evaluation_result, (int, float)):
            return evaluation_result, [], []
        rows = getattr(evaluation_result, "results", []) or []
        outputs = [prediction for _, prediction, _ in rows]
        subscores = [score for _, _, score in rows]
        return evaluation_result.score, outputs, subscores


class GEPA(dspy.teleprompt.teleprompt.Teleprompter):
    def __init__(
        self,
        named_predictor_to_feedback_fn_map: dict[str, callable],
        knowledgebase_qe,
        metric: callable,
        logger,
        run_dir: str,
        run_linearized_gepa: bool=True,
        num_threads=None,
        num_iters=None,
        failure_score=0,
        perfect_score=1,
        teacher_lm: dspy.LM = None,
        use_wandb: bool = False,
        wandb_api_key: str = None,
        max_evals_per_trainval_instance=None,
        seed=0,
        skip_perfect_score=True,
        use_merge=False,
        max_merge_invocations=5,
        num_dspy_examples_per_gepa_step=3,
        max_metric_calls=None,
        max_total_api_calls=None,
        max_total_search_tokens=None,
        api_call_hard_limit=None,
        max_search_iterations=None,
        set_for_merge_minibatch='train',  # 'train', 'val', or 'both'
        track_scores_on: Literal['val', 'train_val'] = 'train_val',
        add_format_failure_as_feedback: bool=False,
        selection_mode: Literal['validation', 'llm_judge', 'llm_judge_score_aware', 'validation_llm_judge_combined']='validation',
        judge_lm: dspy.LM = None,
        judge_learned_guide_path: str = None,
        judge_strict_learned_guide: bool = False,
        judge_memory_top_k: int = 3,
        judge_memory_same_predictor_only: bool = True,
        validation_sidecar_judge_alignment: bool = False,
        always_validate_for_teacher_memory: bool = False,
        combined_score_mode: Literal['normalized', 'direct'] = 'normalized',
        combined_validation_weight: float = 1.0,
        combined_judge_weight: float = 1.0,
        combined_validation_score_scale: float = 100.0,
        combined_min_surrogate_gain: float = 0.01,
        retained_validation_fraction: float = 100.0,
        validation_sampling_mode: str = "fixed",
    ):
        # Exactly one primary stopping constraint should be set.
        assert (
            (max_metric_calls is not None)
            + (max_evals_per_trainval_instance is not None)
            + (num_iters is not None)
            + (max_total_api_calls is not None)
            + (max_total_search_tokens is not None)
            + (max_search_iterations is not None)
        ) == 1, "Exactly one of max_metric_calls, max_evals_per_trainval_instance, num_iters, max_total_api_calls, max_total_search_tokens or max_search_iterations should be set. You set max_metric_calls={}, max_evals_per_trainval_instance={}, num_iters={}, max_total_api_calls={}, max_total_search_tokens={}, max_search_iterations={}".format(
            max_metric_calls, max_evals_per_trainval_instance, num_iters, max_total_api_calls, max_total_search_tokens, max_search_iterations
        )

        self.named_predictor_to_feedback_fn_map = named_predictor_to_feedback_fn_map
        self.knowledgebase_qe = knowledgebase_qe
        self.metric_fn = metric
        self.logger = logger
        self.run_dir = run_dir
        self.run_linearized_gepa = run_linearized_gepa
        self.num_threads = num_threads

        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.teacher_lm = teacher_lm
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key

        # Run constraints
        self.num_iters = num_iters
        self.max_evals_per_trainval_instance = max_evals_per_trainval_instance
        self.max_metric_calls = max_metric_calls
        self.max_total_api_calls = max_total_api_calls
        self.max_total_search_tokens = max_total_search_tokens
        self.api_call_hard_limit = api_call_hard_limit
        self.max_search_iterations = max_search_iterations

        self.seed = seed
        self.skip_perfect_score = skip_perfect_score
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.set_for_merge_minibatch = set_for_merge_minibatch
        if self.set_for_merge_minibatch in ['train', 'both']:
            assert track_scores_on == 'train_val', "track_scores_on should be 'train_val' if set_for_merge_minibatch is 'train' or 'both'. You set track_scores_on={}".format(track_scores_on)

        assert track_scores_on in ['val', 'train_val'], "track_scores_on should be either 'val' or 'train_val'. You set track_scores_on={}".format(track_scores_on)
        self.track_scores_on = track_scores_on
        assert selection_mode in ['validation', 'llm_judge', 'llm_judge_score_aware', 'validation_llm_judge_combined'], f"Unknown selection_mode {selection_mode}"
        self.combined_validation_llm_judge = selection_mode == 'validation_llm_judge_combined'
        self.uses_formal_validation = selection_mode in ['validation', 'validation_llm_judge_combined']
        self.uses_llm_judge = selection_mode in ['llm_judge', 'llm_judge_score_aware', 'validation_llm_judge_combined']
        self.feedback_only_llm_judge = selection_mode in ['llm_judge', 'validation_llm_judge_combined']
        self.score_aware_llm_judge = selection_mode == 'llm_judge_score_aware'
        if self.uses_llm_judge:
            assert self.track_scores_on == 'val', "selection_mode='llm_judge' currently requires track_scores_on='val'"
            assert not self.use_merge, "selection_mode='llm_judge' currently does not support merge mode"
        self.selection_mode = selection_mode
        assert combined_score_mode in ['normalized', 'direct'], f"Unknown combined_score_mode {combined_score_mode}"
        assert combined_validation_score_scale > 0, "combined_validation_score_scale must be positive"
        self.combined_score_mode = combined_score_mode
        self.combined_validation_weight = combined_validation_weight
        self.combined_judge_weight = combined_judge_weight
        self.combined_validation_score_scale = combined_validation_score_scale
        self.combined_min_surrogate_gain = combined_min_surrogate_gain
        self.retained_validation_fraction = retained_validation_fraction
        if validation_sampling_mode != "fixed":
            raise ValueError(
                "validation_sampling_mode='random_per_iteration' has been removed. "
                "Retained validation examples must be sampled once before GEPA starts and kept fixed."
            )
        self.validation_sampling_mode = validation_sampling_mode
        self.judge_lm = judge_lm
        self.judge_learned_guide_path = judge_learned_guide_path
        self._learned_judge_guide_text = None
        self.judge_strict_learned_guide = judge_strict_learned_guide
        if self.judge_strict_learned_guide:
            assert self.judge_learned_guide_path, "judge_strict_learned_guide requires judge_learned_guide_path"
        self.judge_memory_top_k = judge_memory_top_k
        self.judge_memory_same_predictor_only = judge_memory_same_predictor_only
        self.judge_memory_bank_path = os.path.join(run_dir, "judge_memory_bank.jsonl")
        self.judge_memory_records = None
        self.validation_sidecar_judge_alignment = validation_sidecar_judge_alignment
        if self.validation_sidecar_judge_alignment:
            assert self.selection_mode == 'validation', "validation_sidecar_judge_alignment only applies to validation warmup runs"
        self.always_validate_for_teacher_memory = always_validate_for_teacher_memory
        if self.always_validate_for_teacher_memory:
            assert self.selection_mode == 'validation', "always_validate_for_teacher_memory only applies to validation warmup runs"
        self.judge_alignment_memory_bank_path = os.path.join(run_dir, "judge_alignment_memory_bank.jsonl")
        self.judge_alignment_memory_records = None

        self.valset_provided = None
        self.train_val_size = None

        self.num_dspy_examples_per_gepa_step = num_dspy_examples_per_gepa_step

        self.add_format_failure_as_feedback = add_format_failure_as_feedback

        self.shuffled_trainset_ids = []
        self.epoch = -1
        self.id_freqs = Counter()
        self.gepa_state = None

    def compile(
        self, student, trainset, valset,
    ):
        if valset is not None:
            self.valset_provided = True

        assert trainset is not None, "Trainset must be provided"

        gepa_state = self.gepa(
            base_dspy_program=student,
            trainset=trainset,
            valset=valset,
        )

        best_prog_idx = idxmax(gepa_state.per_program_tracked_scores)
        best_prog = gepa_state.program_candidates[best_prog_idx]
        return best_prog

    def update_shuffled_trainset(self, original_trainset):
        self.shuffled_trainset_ids = list(range(len(original_trainset)))
        self.gepa_state.rng1.shuffle(self.shuffled_trainset_ids)
        for id in self.shuffled_trainset_ids:
            self.id_freqs[id] += 1

        num_to_pad = self.num_dspy_examples_per_gepa_step - (len(original_trainset) % self.num_dspy_examples_per_gepa_step)
        if num_to_pad > 0:
            # Select ids based on least frequent ids
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_trainset_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def select_training_sample_and_update_shuffled_trainset(
        self,
        original_trainset: List[Example],
        train_step_idx: int,
    ) -> List[Example]:
        base_idx = train_step_idx * self.num_dspy_examples_per_gepa_step
        if self.epoch == -1:
            curr_epoch = 0
        else:
            curr_epoch = base_idx // len(self.shuffled_trainset_ids)
        if curr_epoch > self.epoch:
            print(f"Updating shuffled trainset for epoch {curr_epoch}...")
            self.epoch = curr_epoch
            self.update_shuffled_trainset(original_trainset)

        assert len(self.shuffled_trainset_ids) >= self.num_dspy_examples_per_gepa_step, f"Shuffled trainset length {len(self.shuffled_trainset_ids)} is less than num_dspy_examples_per_grpo_step {self.num_dspy_examples_per_gepa_step}"
        assert len(self.shuffled_trainset_ids) % self.num_dspy_examples_per_gepa_step == 0, f"Shuffled trainset length {len(self.shuffled_trainset_ids)} is not divisible by num_dspy_examples_per_grpo_step {self.num_dspy_examples_per_gepa_step}"

        base_idx = base_idx % len(self.shuffled_trainset_ids)
        end_idx = base_idx + self.num_dspy_examples_per_gepa_step
        assert end_idx <= len(self.shuffled_trainset_ids), f"End index {end_idx} is out of bounds for shuffled trainset length {len(self.shuffled_trainset_ids)}"
        selected_ids = self.shuffled_trainset_ids[base_idx:end_idx]
        return selected_ids

    def select_eval_subsample_for_merged_program(
        self,
        scores1,
        scores2,
        rng: random.Random,
        num_subsample_ids: int = 5,
    ) -> List[int]:
        all_indices = set(range(len(scores1)))
        # Partitioning
        partition1_ids = [i for i, (s1, s2) in enumerate(zip(scores1, scores2)) if s1 > s2]
        partition2_ids = [i for i, (s1, s2) in enumerate(zip(scores1, scores2)) if s2 > s1]
        partition3_ids = [i for i in all_indices if i not in partition1_ids and i not in partition2_ids]
        # Set up sample sizes
        n_each = math.ceil(num_subsample_ids / 3)
        n1 = min(len(partition1_ids), n_each)
        n2 = min(len(partition2_ids), n_each)
        n3 = min(len(partition3_ids), num_subsample_ids - (n1 + n2))
        # Sample
        # rng = gepa_state.rng1
        selected = []
        if n1: selected += rng.sample(partition1_ids, k=n1)
        if n2: selected += rng.sample(partition2_ids, k=n2)
        if n3: selected += rng.sample(partition3_ids, k=n3)
        # Pad if needed to desired length, without duplicates if possible
        remaining = num_subsample_ids - len(selected)
        unused = list(all_indices - set(selected))
        if remaining > 0:
            if len(unused) >= remaining:
                selected += rng.sample(unused, k=remaining)
            else:
                # All unique exhausted; use replacement
                selected += rng.choices(list(all_indices), k=remaining)
        return selected[:num_subsample_ids]

    def get_pareto_front_programs(self, gepa_state: GEPAState) -> List[Set[int]]:
        return (
            gepa_state.program_at_pareto_front_valset + \
            gepa_state.program_at_pareto_front + \
            [
                # {idxmax(train_val_weighted_agg_scores_for_all_programs)}, # Add best aggregate program
                # {idxmax(gepa_state.program_full_scores_val_set)}, # Add best on valset
                # {idxmax(gepa_state.program_full_scores)}, # Add best on trainset
            ] # TODO: Think about whether this should be added or not. Make this configurable.
        ) if self.track_scores_on == 'train_val' else gepa_state.program_at_pareto_front_valset

    def select_next_candidate_to_update(self, gepa_state: GEPAState):
        # TODO: Update this method to use pareto front from both train and val sets configurable
        assert len(gepa_state.per_program_tracked_scores) == len(gepa_state.program_candidates)

        if not gepa_state.running_linearized_gepa:
            curr_prog_id = select_program_candidate_from_pareto_front(
                self.get_pareto_front_programs(gepa_state),
                gepa_state.per_program_tracked_scores,
                gepa_state.rng1,
            )
        else:
            curr_prog_id = idxmax(gepa_state.per_program_tracked_scores)

        return curr_prog_id

    def append_jsonl_record(self, filename: str, payload: dict):
        with open(os.path.join(self.run_dir, filename), 'a') as f:
            f.write(json.dumps(payload, default=lambda x: {**x} if hasattr(x, "keys") else repr(x)) + "\n")

    def get_lm_usage_snapshot(self, lm):
        if lm is None:
            return {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
        traces = getattr(lm, "history", [])
        return {
            "input_tokens": sum((trace.get("usage", {}) or {}).get("prompt_tokens", 0) for trace in traces),
            "output_tokens": sum((trace.get("usage", {}) or {}).get("completion_tokens", 0) for trace in traces),
            "api_calls": len(traces),
        }

    def increment_validation_usage_from_lm(self, gepa_state: GEPAState, lm, usage_before: dict):
        usage_after = self.get_lm_usage_snapshot(lm)
        gepa_state.validation_input_tokens += usage_after["input_tokens"] - usage_before["input_tokens"]
        gepa_state.validation_output_tokens += usage_after["output_tokens"] - usage_before["output_tokens"]
        gepa_state.validation_api_calls += usage_after["api_calls"] - usage_before["api_calls"]

    def increment_minibatch_usage_from_lm(self, gepa_state: GEPAState, lm, usage_before: dict):
        usage_after = self.get_lm_usage_snapshot(lm)
        gepa_state.minibatch_input_tokens += usage_after["input_tokens"] - usage_before["input_tokens"]
        gepa_state.minibatch_output_tokens += usage_after["output_tokens"] - usage_before["output_tokens"]
        gepa_state.minibatch_api_calls += usage_after["api_calls"] - usage_before["api_calls"]

    def get_total_search_api_calls(self):
        optimizer_usage = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm)
        judge_usage = self.get_lm_usage_snapshot(self.judge_lm)
        return optimizer_usage["api_calls"] + judge_usage["api_calls"]

    def get_total_search_tokens(self):
        optimizer_usage = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm)
        judge_usage = self.get_lm_usage_snapshot(self.judge_lm)
        return (
            optimizer_usage["input_tokens"]
            + optimizer_usage["output_tokens"]
            + judge_usage["input_tokens"]
            + judge_usage["output_tokens"]
        )

    def build_cost_accounting_payload(self, optimizer_usage: dict, judge_usage: dict, gepa_state: GEPAState):
        validation_input_tokens = getattr(gepa_state, "validation_input_tokens", 0)
        validation_output_tokens = getattr(gepa_state, "validation_output_tokens", 0)
        validation_tokens = validation_input_tokens + validation_output_tokens
        minibatch_input_tokens = getattr(gepa_state, "minibatch_input_tokens", 0)
        minibatch_output_tokens = getattr(gepa_state, "minibatch_output_tokens", 0)
        minibatch_tokens = minibatch_input_tokens + minibatch_output_tokens
        optimizer_total_tokens = optimizer_usage["input_tokens"] + optimizer_usage["output_tokens"]
        judge_total_tokens = judge_usage["input_tokens"] + judge_usage["output_tokens"]
        optimizer_overhead_tokens = max(0, optimizer_total_tokens - validation_tokens - minibatch_tokens)
        optimizer_overhead_api_calls = max(
            0,
            optimizer_usage["api_calls"]
            - getattr(gepa_state, "validation_api_calls", 0)
            - getattr(gepa_state, "minibatch_api_calls", 0),
        )

        return {
            "minibatch_input_tokens": minibatch_input_tokens,
            "minibatch_output_tokens": minibatch_output_tokens,
            "minibatch_tokens": minibatch_tokens,
            "minibatch_api_calls": getattr(gepa_state, "minibatch_api_calls", 0),
            "optimizer_overhead_tokens": optimizer_overhead_tokens,
            "optimizer_overhead_api_calls": optimizer_overhead_api_calls,
            "optimization_control_tokens": optimizer_overhead_tokens + judge_total_tokens,
            "optimization_control_api_calls": optimizer_overhead_api_calls + judge_usage["api_calls"],
            "search_total_tokens": optimizer_total_tokens + judge_total_tokens,
            "search_total_api_calls": optimizer_usage["api_calls"] + judge_usage["api_calls"],
            "system_execution_tokens": validation_tokens + minibatch_tokens,
            "system_execution_api_calls": getattr(gepa_state, "validation_api_calls", 0) + getattr(gepa_state, "minibatch_api_calls", 0),
        }

    def load_learned_judge_guide(self):
        if self._learned_judge_guide_text is not None:
            return self._learned_judge_guide_text
        if not self.judge_learned_guide_path:
            self._learned_judge_guide_text = None
            return None
        if not os.path.exists(self.judge_learned_guide_path):
            self.logger.log(f"Learned judge guide path does not exist: {self.judge_learned_guide_path}")
            self._learned_judge_guide_text = None
            return None
        with open(self.judge_learned_guide_path, "r", encoding="utf-8") as f:
            self._learned_judge_guide_text = f.read().strip()
        return self._learned_judge_guide_text

    def load_judge_memory_records(self):
        if self.judge_memory_records is None:
            self.judge_memory_records = load_memory_bank(self.judge_memory_bank_path)
        return self.judge_memory_records

    def load_alignment_memory_records(self):
        if self.judge_alignment_memory_records is None:
            self.judge_alignment_memory_records = load_memory_bank(self.judge_alignment_memory_bank_path)
        return self.judge_alignment_memory_records

    def build_teacher_correction_text(self, student_preferred_prompt: str, teacher_preferred: str, teacher_delta: float | None):
        if student_preferred_prompt == teacher_preferred:
            return f"Student and validation teacher agreed on `{teacher_preferred}`."
        delta_note = ""
        if teacher_delta is not None:
            delta_note = f" Validation delta={teacher_delta}."
        return (
            f"Student preferred `{student_preferred_prompt}`, but validation teacher preferred `{teacher_preferred}`."
            f"{delta_note}"
        )

    def compute_combined_selection_metadata(self, old_validation_score, new_validation_score, judge_decision: dict):
        validation_delta = float(new_validation_score) - float(old_validation_score)
        if self.combined_score_mode == 'normalized':
            validation_component = validation_delta / self.combined_validation_score_scale
        else:
            validation_component = validation_delta

        judge_confidence = float(judge_decision.get("confidence", 0.0) or 0.0)
        judge_signed_confidence = judge_confidence if judge_decision.get("preferred_prompt") == "new" else -judge_confidence
        judge_component = judge_signed_confidence

        weighted_validation_component = self.combined_validation_weight * validation_component
        weighted_judge_component = self.combined_judge_weight * judge_component
        combined_delta = weighted_validation_component + weighted_judge_component
        teacher_preferred = teacher_preferred_prompt(old_validation_score, new_validation_score)
        combined_accept = combined_delta > 0

        return {
            "combined_score_mode": self.combined_score_mode,
            "combined_validation_weight": self.combined_validation_weight,
            "combined_judge_weight": self.combined_judge_weight,
            "combined_validation_score_scale": self.combined_validation_score_scale,
            "old_validation_score": old_validation_score,
            "new_validation_score": new_validation_score,
            "validation_delta": validation_delta,
            "validation_component": validation_component,
            "weighted_validation_component": weighted_validation_component,
            "judge_confidence": judge_confidence,
            "judge_signed_confidence": judge_signed_confidence,
            "judge_component": judge_component,
            "weighted_judge_component": weighted_judge_component,
            "combined_delta": combined_delta,
            "combined_accept": combined_accept,
            "validation_teacher_preferred_prompt": teacher_preferred,
        }

    def get_combined_validation_score_for_program(self, gepa_state: GEPAState, program_idx: int):
        raw_scores = getattr(gepa_state, "combined_validation_full_scores", None)
        if raw_scores is not None and program_idx < len(raw_scores):
            return raw_scores[program_idx]
        return None

    def append_combined_validation_score_to_state(
        self,
        gepa_state: GEPAState,
        new_program_idx: int,
        validation_score,
        validation_subscores,
    ):
        if not hasattr(gepa_state, "combined_validation_full_scores") or gepa_state.combined_validation_full_scores is None:
            gepa_state.combined_validation_full_scores = [None] * new_program_idx
        if not hasattr(gepa_state, "combined_validation_val_subscores") or gepa_state.combined_validation_val_subscores is None:
            gepa_state.combined_validation_val_subscores = [None] * new_program_idx

        while len(gepa_state.combined_validation_full_scores) < new_program_idx:
            gepa_state.combined_validation_full_scores.append(None)
        while len(gepa_state.combined_validation_val_subscores) < new_program_idx:
            gepa_state.combined_validation_val_subscores.append(None)

        if len(gepa_state.combined_validation_full_scores) == new_program_idx:
            gepa_state.combined_validation_full_scores.append(validation_score)
        else:
            gepa_state.combined_validation_full_scores[new_program_idx] = validation_score

        validation_subscores = list(validation_subscores) if validation_subscores is not None else None
        if len(gepa_state.combined_validation_val_subscores) == new_program_idx:
            gepa_state.combined_validation_val_subscores.append(validation_subscores)
        else:
            gepa_state.combined_validation_val_subscores[new_program_idx] = validation_subscores

    def append_combined_validation_memory_records(
        self,
        gepa_state: GEPAState,
        selected_program_candidate: int,
        new_program_idx: int | None,
        predictor_name: str,
        old_instruction: str,
        new_instruction: str,
        dataset_with_feedback: list[dict],
        old_validation_score,
        new_validation_score,
        judge_decision: dict,
        combined_metadata: dict,
        retrieved_alignment_records: list[dict] | None = None,
    ):
        teacher_delta = combined_metadata.get("validation_delta")
        teacher_preferred = combined_metadata.get("validation_teacher_preferred_prompt")
        student_preferred = judge_decision.get("preferred_prompt", "old")
        ranking_match = student_preferred == teacher_preferred
        record_suffix = f"prog_{new_program_idx}" if new_program_idx is not None else "rejected"

        teacher_record = {
            "memory_record_id": f"iter_{gepa_state.i + 1}_{record_suffix}_combined_teacher",
            "record_type": "combined_validation_teacher_v1",
            "iteration": gepa_state.i + 1,
            "selected_program_candidate": selected_program_candidate,
            "new_program_idx": new_program_idx,
            "predictor_name": predictor_name,
            "old_instruction": old_instruction,
            "new_instruction": new_instruction,
            "feedback_example_count": len(dataset_with_feedback),
            "reflective_evidence_summary": summarize_feedback_examples(dataset_with_feedback, include_scores=False),
            "teacher_old_val_score": old_validation_score,
            "teacher_new_val_score": new_validation_score,
            "teacher_delta": teacher_delta,
            "teacher_delta_bucket": delta_bucket(teacher_delta),
            "teacher_label": teacher_label_from_scores(old_validation_score, new_validation_score),
            "teacher_preferred_prompt": teacher_preferred,
            "student_preferred_prompt": student_preferred,
            "student_confidence": judge_decision.get("confidence"),
            "student_signed_confidence": combined_metadata.get("judge_signed_confidence"),
            "student_component": combined_metadata.get("judge_component"),
            "ranking_match": ranking_match,
            "combined_delta": combined_metadata.get("combined_delta"),
            "combined_accept": combined_metadata.get("combined_accept"),
            "selection_mode": self.selection_mode,
        }
        self.append_jsonl_record("judge_memory_bank.jsonl", teacher_record)
        if self.judge_memory_records is not None:
            self.judge_memory_records.append(teacher_record)

        alignment_record = {
            "alignment_record_id": f"iter_{gepa_state.i + 1}_{record_suffix}_combined_alignment",
            "record_type": "combined_teacher_student_alignment_v1",
            "iteration": gepa_state.i + 1,
            "selected_program_candidate": selected_program_candidate,
            "new_program_idx": new_program_idx,
            "predictor_name": predictor_name,
            "old_instruction": old_instruction,
            "new_instruction": new_instruction,
            "feedback_example_count": len(dataset_with_feedback),
            "reflective_evidence_summary": summarize_feedback_examples(dataset_with_feedback, include_scores=False),
            "teacher_old_val_score": old_validation_score,
            "teacher_new_val_score": new_validation_score,
            "teacher_delta": teacher_delta,
            "teacher_delta_bucket": delta_bucket(teacher_delta),
            "teacher_label": teacher_label_from_scores(old_validation_score, new_validation_score),
            "teacher_preferred_prompt": teacher_preferred,
            "student_preferred_prompt": student_preferred,
            "student_component": combined_metadata.get("judge_component"),
            "student_confidence": judge_decision.get("confidence"),
            "student_signed_confidence": combined_metadata.get("judge_signed_confidence"),
            "student_parse_status": judge_decision.get("parse_status"),
            "ranking_match": ranking_match,
            "correction_direction": "aligned" if ranking_match else f"student_{student_preferred}_teacher_{teacher_preferred}",
            "teacher_correction": self.build_teacher_correction_text(student_preferred, teacher_preferred, teacher_delta),
            "combined_delta": combined_metadata.get("combined_delta"),
            "combined_accept": combined_metadata.get("combined_accept"),
            "retrieved_alignment_record_ids": [
                record.get("alignment_record_id") or record.get("memory_record_id")
                for record in (retrieved_alignment_records or [])
            ],
            "selection_mode": self.selection_mode,
        }
        self.append_jsonl_record("judge_alignment_memory_bank.jsonl", alignment_record)
        if self.judge_alignment_memory_records is not None:
            self.judge_alignment_memory_records.append(alignment_record)
        return teacher_record, alignment_record

    def append_validation_teacher_memory_record(
        self,
        gepa_state: GEPAState,
        selected_program_candidate: int,
        new_program_idx: int,
        predictor_name: str,
        old_instruction: str,
        new_instruction: str,
        dataset_with_feedback: list[dict],
        old_subsample_score,
        new_subsample_score,
    ):
        old_val_score = gepa_state.program_full_scores_val_set[selected_program_candidate]
        new_val_score = gepa_state.program_full_scores_val_set[new_program_idx]
        teacher_delta = None if old_val_score is None or new_val_score is None else round(new_val_score - old_val_score, 6)
        record = {
            "memory_record_id": f"iter_{gepa_state.i + 1}_prog_{new_program_idx}",
            "iteration": gepa_state.i + 1,
            "selected_program_candidate": selected_program_candidate,
            "new_program_idx": new_program_idx,
            "predictor_name": predictor_name,
            "old_instruction": old_instruction,
            "new_instruction": new_instruction,
            "feedback_example_count": len(dataset_with_feedback),
            "reflective_evidence_summary": summarize_feedback_examples(dataset_with_feedback, include_scores=False),
            "old_subsample_score": old_subsample_score,
            "new_subsample_score": new_subsample_score,
            "teacher_old_val_score": old_val_score,
            "teacher_new_val_score": new_val_score,
            "teacher_delta": teacher_delta,
            "teacher_delta_bucket": delta_bucket(teacher_delta),
            "teacher_label": teacher_label_from_scores(old_val_score, new_val_score),
            "teacher_preferred_prompt": teacher_preferred_prompt(old_val_score, new_val_score),
            "selection_mode": self.selection_mode,
        }
        self.append_jsonl_record("judge_memory_bank.jsonl", record)
        if self.judge_memory_records is not None:
            self.judge_memory_records.append(record)

    def append_validation_teacher_memory_record_from_scores(
        self,
        gepa_state: GEPAState,
        selected_program_candidate: int,
        predictor_name: str,
        old_instruction: str,
        new_instruction: str,
        dataset_with_feedback: list[dict],
        old_subsample_score,
        new_subsample_score,
        old_validation_score,
        new_validation_score,
        record_suffix: str,
    ):
        teacher_delta = None if old_validation_score is None or new_validation_score is None else round(new_validation_score - old_validation_score, 6)
        record = {
            "memory_record_id": f"iter_{gepa_state.i + 1}_{record_suffix}",
            "record_type": "validation_teacher_pair_v1",
            "iteration": gepa_state.i + 1,
            "selected_program_candidate": selected_program_candidate,
            "new_program_idx": None,
            "predictor_name": predictor_name,
            "old_instruction": old_instruction,
            "new_instruction": new_instruction,
            "feedback_example_count": len(dataset_with_feedback),
            "reflective_evidence_summary": summarize_feedback_examples(dataset_with_feedback, include_scores=False),
            "old_subsample_score": old_subsample_score,
            "new_subsample_score": new_subsample_score,
            "teacher_old_val_score": old_validation_score,
            "teacher_new_val_score": new_validation_score,
            "teacher_delta": teacher_delta,
            "teacher_delta_bucket": delta_bucket(teacher_delta),
            "teacher_label": teacher_label_from_scores(old_validation_score, new_validation_score),
            "teacher_preferred_prompt": teacher_preferred_prompt(old_validation_score, new_validation_score),
            "selection_mode": self.selection_mode,
            "teacher_memory_only": True,
        }
        self.append_jsonl_record("judge_memory_bank.jsonl", record)
        if self.judge_memory_records is not None:
            self.judge_memory_records.append(record)

    def append_validation_alignment_memory_record(
        self,
        gepa_state: GEPAState,
        selected_program_candidate: int,
        new_program_idx: int,
        predictor_name: str,
        old_instruction: str,
        new_instruction: str,
        dataset_with_feedback: list[dict],
        student_prediction: dict,
        retrieved_alignment_records: list[dict] | None = None,
    ):
        old_val_score = gepa_state.program_full_scores_val_set[selected_program_candidate]
        new_val_score = gepa_state.program_full_scores_val_set[new_program_idx]
        teacher_delta = None if old_val_score is None or new_val_score is None else round(new_val_score - old_val_score, 6)
        teacher_preferred = teacher_preferred_prompt(old_val_score, new_val_score)
        student_preferred = student_prediction.get("preferred_prompt", "old")
        ranking_match = student_preferred == teacher_preferred
        correction_direction = "aligned" if ranking_match else f"student_{student_preferred}_teacher_{teacher_preferred}"
        record = {
            "alignment_record_id": f"iter_{gepa_state.i + 1}_prog_{new_program_idx}_alignment",
            "record_type": "teacher_student_alignment_v1",
            "iteration": gepa_state.i + 1,
            "selected_program_candidate": selected_program_candidate,
            "new_program_idx": new_program_idx,
            "predictor_name": predictor_name,
            "old_instruction": old_instruction,
            "new_instruction": new_instruction,
            "feedback_example_count": len(dataset_with_feedback),
            "reflective_evidence_summary": summarize_feedback_examples(dataset_with_feedback, include_scores=False),
            "teacher_old_val_score": old_val_score,
            "teacher_new_val_score": new_val_score,
            "teacher_delta": teacher_delta,
            "teacher_delta_bucket": delta_bucket(teacher_delta),
            "teacher_label": teacher_label_from_scores(old_val_score, new_val_score),
            "teacher_preferred_prompt": teacher_preferred,
            "student_old_pairwise_score": student_prediction.get("old_score"),
            "student_new_pairwise_score": student_prediction.get("new_score"),
            "student_preferred_prompt": student_preferred,
            "student_confidence": student_prediction.get("confidence"),
            "student_short_reason": student_prediction.get("short_reason"),
            "student_risk_note": student_prediction.get("risk_note"),
            "student_parse_status": student_prediction.get("parse_status"),
            "student_prompt_version": student_prediction.get("prompt_version"),
            "student_used_alignment_memory": student_prediction.get("used_alignment_memory"),
            "ranking_match": ranking_match,
            "correction_direction": correction_direction,
            "teacher_correction": self.build_teacher_correction_text(student_preferred, teacher_preferred, teacher_delta),
            "retrieved_alignment_record_ids": [
                record.get("alignment_record_id") or record.get("memory_record_id")
                for record in (retrieved_alignment_records or [])
            ],
            "selection_mode": self.selection_mode,
        }
        if not ranking_match:
            self.append_jsonl_record("judge_alignment_memory_bank.jsonl", record)
            if self.judge_alignment_memory_records is not None:
                self.judge_alignment_memory_records.append(record)
        return record

    def append_judge_memory_retrieval_record(
        self,
        gepa_state: GEPAState,
        predictor_name: str,
        selected_program_candidate: int,
        retrieved_records: list[dict],
        retrieved_alignment_records: list[dict] | None = None,
    ):
        payload = {
            "iteration": gepa_state.i + 1,
            "predictor_name": predictor_name,
            "selected_program_candidate": selected_program_candidate,
            "retrieved_memory_record_ids": [record.get("memory_record_id") for record in retrieved_records],
            "retrieved_teacher_labels": [record.get("teacher_label") for record in retrieved_records],
            "retrieved_teacher_delta_buckets": [record.get("teacher_delta_bucket") for record in retrieved_records],
            "retrieved_alignment_record_ids": [
                record.get("alignment_record_id") or record.get("memory_record_id")
                for record in (retrieved_alignment_records or [])
            ],
            "retrieved_alignment_corrections": [
                record.get("teacher_correction") or record.get("correction_direction")
                for record in (retrieved_alignment_records or [])
            ],
        }
        self.append_jsonl_record("judge_memory_retrievals.jsonl", payload)

    def append_metric_call_checkpoint(
        self,
        gepa_state: GEPAState,
        phase: str,
        metric_calls_before: int,
        metric_calls_after: int,
        extra: dict | None = None,
    ):
        payload = {
            "iteration": gepa_state.i + 1,
            "phase": phase,
            "selection_mode": self.selection_mode,
            "metric_calls_before": metric_calls_before,
            "metric_calls_after": metric_calls_after,
            "best_program_idx": idxmax(gepa_state.per_program_tracked_scores),
            "best_tracked_score": max(gepa_state.per_program_tracked_scores),
            "num_candidates": len(gepa_state.program_candidates),
            "accepted_updates_so_far": max(0, len(gepa_state.program_candidates) - 1),
            "selected_program_candidate": gepa_state.full_program_trace[-1].get("selected_program_candidate") if gepa_state.full_program_trace else None,
            "predictor_name_to_update": gepa_state.full_program_trace[-1].get("predictor_name_to_update") if gepa_state.full_program_trace else None,
            "new_program_idx": gepa_state.full_program_trace[-1].get("new_program_idx") if gepa_state.full_program_trace else None,
            "validation_input_tokens": gepa_state.validation_input_tokens,
            "validation_output_tokens": gepa_state.validation_output_tokens,
            "validation_tokens": gepa_state.validation_input_tokens + gepa_state.validation_output_tokens,
            "validation_api_calls": gepa_state.validation_api_calls,
        }
        optimizer_usage = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm)
        judge_usage = self.get_lm_usage_snapshot(self.judge_lm)
        payload.update({
            "optimizer_input_tokens": optimizer_usage["input_tokens"],
            "optimizer_output_tokens": optimizer_usage["output_tokens"],
            "optimizer_total_tokens": optimizer_usage["input_tokens"] + optimizer_usage["output_tokens"],
            "optimizer_api_calls": optimizer_usage["api_calls"],
            "judge_input_tokens": judge_usage["input_tokens"],
            "judge_output_tokens": judge_usage["output_tokens"],
            "judge_total_tokens": judge_usage["input_tokens"] + judge_usage["output_tokens"],
            "judge_api_calls": judge_usage["api_calls"],
            "optimizer_elapsed_seconds": time.perf_counter() - self.optimizer_phase_start_time if getattr(self, "optimizer_phase_start_time", None) is not None else None,
        })
        payload.update(self.build_cost_accounting_payload(optimizer_usage, judge_usage, gepa_state))
        if extra:
            payload.update(extra)
        self.append_jsonl_record("metric_call_checkpoints.jsonl", payload)

    def run_full_eval_add_new_program_to_gepa_tree(
        self,
        new_program: dspy.Module,
        gepa_state: GEPAState,
        trainset_evaluator: dspy.Evaluate,
        valset_evaluator: dspy.Evaluate,
        parent_program_idx: List[int]
    ):
        num_metric_calls_by_discovery_of_new_program = gepa_state.total_num_evals
        validation_usage_before = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm or new_program.get_lm())

        # Calculate metrics for new program and update gepa state
        with lm_trace_context(
            "validation_full_eval",
            iteration=gepa_state.i + 1,
            parent_program_idx=parent_program_idx,
        ):
            if self.track_scores_on == 'train_val':
                trainset_score, trainset_outputs, trainset_subscores = trainset_evaluator(new_program)
            else:
                assert self.track_scores_on == 'val', "track_scores_on should be either 'val' or 'train_val'. You set track_scores_on={}".format(self.track_scores_on)
                trainset_score, trainset_outputs, trainset_subscores = None, None, None
            valset_score, valset_outputs, valset_subscores = valset_evaluator(new_program)
        self.increment_validation_usage_from_lm(gepa_state, dspy.dsp.utils.settings.lm or new_program.get_lm(), validation_usage_before)

        # We have run one full eval of the new program on train set and val set
        gepa_state.num_full_ds_evals += 1
        if self.track_scores_on == 'train_val':
            gepa_state.total_num_evals_per_trainval_instance += 1
        else:
            assert self.track_scores_on == 'val', "track_scores_on should be either 'val' or 'train_val'. You set track_scores_on={}".format(self.track_scores_on)
            gepa_state.total_num_evals_per_trainval_instance += (len(valset_subscores) / self.train_val_size)
        gepa_state.total_num_evals += len(trainset_subscores) + len(valset_subscores) if self.track_scores_on == 'train_val' else len(valset_subscores)

        new_program_idx, linear_pareto_front_program_idx = gepa_state.update_state_with_new_program(
            parent_program_idx=parent_program_idx, # TODO: Handle this better. Mark both parents
            new_program=new_program,
            trainset_score=trainset_score,
            trainset_outputs=trainset_outputs,
            trainset_subscores=trainset_subscores,
            valset_score=valset_score,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            run_dir=self.run_dir,
            track_scores_on=self.track_scores_on,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery_of_new_program
        )

        gepa_state.full_program_trace[-1]['new_program_idx'] = new_program_idx
        self.append_metric_call_checkpoint(
            gepa_state=gepa_state,
            phase="full_validation_eval",
            metric_calls_before=num_metric_calls_by_discovery_of_new_program,
            metric_calls_after=gepa_state.total_num_evals,
            extra={
                "new_program_idx": new_program_idx,
                "linear_pareto_front_program_idx": linear_pareto_front_program_idx,
            },
        )

        if new_program_idx == linear_pareto_front_program_idx:
            self.logger.log(f"Iteration {gepa_state.i+1}: New program is on the linear pareto front")

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=gepa_state,
            valset_score=valset_score,
            new_prog_all_scores=trainset_subscores,
            full_score=trainset_score,
            new_program_idx=new_program_idx,
            valset_subscores=valset_subscores,
            new_instruction="Merged program",
            use_wandb=self.use_wandb,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
            track_scores_on= self.track_scores_on,
            selection_mode=self.selection_mode,
        )

        return new_program_idx, linear_pareto_front_program_idx

    def run_surrogate_selection_add_new_program_to_gepa_tree(
        self,
        new_program: dspy.Module,
        gepa_state: GEPAState,
        parent_program_idx: List[int],
        surrogate_score: float,
        selection_metadata: dict,
        new_instruction: str,
    ):
        num_metric_calls_by_discovery_of_new_program = gepa_state.total_num_evals
        surrogate_eval_output = make_selection_surrogate_eval_output(
            score=surrogate_score,
            metadata={
                "selection_mode": self.selection_mode,
                "selection_metadata": selection_metadata,
            },
        )

        new_program_idx, linear_pareto_front_program_idx = gepa_state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            trainset_score=None,
            trainset_outputs=None,
            trainset_subscores=None,
            valset_score=surrogate_eval_output[0],
            valset_outputs=surrogate_eval_output[1],
            valset_subscores=surrogate_eval_output[2],
            run_dir=self.run_dir,
            track_scores_on=self.track_scores_on,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery_of_new_program,
        )

        gepa_state.full_program_trace[-1]['new_program_idx'] = new_program_idx
        gepa_state.full_program_trace[-1]['selection_surrogate_score'] = surrogate_score
        gepa_state.full_program_trace[-1]['selection_mode'] = self.selection_mode

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=gepa_state,
            valset_score=surrogate_eval_output[0],
            new_prog_all_scores=None,
            full_score=None,
            new_program_idx=new_program_idx,
            valset_subscores=surrogate_eval_output[2],
            new_instruction=new_instruction,
            use_wandb=self.use_wandb,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
            track_scores_on=self.track_scores_on,
            selection_mode=self.selection_mode,
            selection_metadata=selection_metadata,
        )

        return new_program_idx, linear_pareto_front_program_idx

    def gepa(
        self,
        base_dspy_program: dspy.Module,
        trainset: list[dspy.Example],
        valset: list[dspy.Example]=None,
        gepa_state_to_use: Union[GEPAState, None]=None,
    ):
        if self.use_wandb:
            wandb_run = initialize_wandb(wandb_api_key=self.wandb_api_key, run_dir=self.run_dir)

        if self.num_threads is None:
            self.num_threads = os.cpu_count()

        trainset_evaluator = LegacyEvaluate(
            devset=trainset,
            metric=self.metric_fn,
            num_threads=self.num_threads,
            return_all_scores=True,
            return_outputs=True,
            failure_score=self.failure_score,
            provide_traceback=True,
            max_errors=len(trainset) * 100  # Allow for many errors in the training set
        )

        if self.uses_formal_validation:
            if valset is None:
                valset = trainset

            valset_evaluator = LegacyEvaluate(
                devset=valset,
                metric=self.metric_fn,
                num_threads=self.num_threads,
                return_all_scores=True,
                return_outputs=True,
                failure_score=self.failure_score,
                provide_traceback=True,
                max_errors=len(valset) * 100  # upper bound from full set
            )
            self.train_val_size = len(trainset) + len(valset)
        else:
            valset_evaluator = None
            self.train_val_size = len(trainset)

        gepa_state = initialize_gepa_state(
            gepa_state_to_use=gepa_state_to_use,
            run_dir=self.run_dir,
            logger=self.logger,
            base_dspy_program=base_dspy_program,
            trainset_evaluator=trainset_evaluator,
            valset_evaluator=valset_evaluator,
            seed=self.seed,
            run_linearized_gepa=self.run_linearized_gepa,
            track_scores_on=self.track_scores_on,
            train_val_size= self.train_val_size,
            selection_mode=self.selection_mode,
        )

        self.gepa_state = gepa_state

        if self.track_scores_on == 'train_val':
            assert len(gepa_state.pareto_front) == len(trainset)

        if self.selection_mode == 'validation':
            expected_valset_len = len(valset_evaluator.devset) if valset_evaluator is not None else len(valset)
            assert len(gepa_state.pareto_front_valset) == expected_valset_len, f"Pareto front valset length {len(gepa_state.pareto_front_valset)} does not match valset length {expected_valset_len}"
        else:
            assert len(gepa_state.pareto_front_valset) == 1, f"Surrogate frontier should contain exactly one tracked score, found {len(gepa_state.pareto_front_valset)}"

        if self.use_wandb:
            # assert gepa_state.i + 1 == 0
            if self.selection_mode == 'validation':
                wandb.log({
                    "base_program_full_trainset_score": gepa_state.program_full_scores[0] if self.track_scores_on == 'train_val' else None,
                    "base_program_full_valset_score": gepa_state.program_full_scores_val_set[0],
                    "iteration": gepa_state.i+1,
                })
            else:
                wandb.log({
                    "base_program_selection_surrogate_score": gepa_state.program_full_scores_val_set[0],
                    "base_program_combined_validation_score": self.get_combined_validation_score_for_program(gepa_state, 0),
                    "iteration": gepa_state.i+1,
                })
        if self.track_scores_on == 'train_val':
            self.logger.log(f"Iteration {gepa_state.i+1}: Base program full trainset score: {gepa_state.program_full_scores[0]}")
        if self.selection_mode == 'validation':
            self.logger.log(f"Iteration {gepa_state.i+1}: Base program full valset score: {gepa_state.program_full_scores_val_set[0]}")
        else:
            self.logger.log(f"Iteration {gepa_state.i+1}: Base program selection surrogate score: {gepa_state.program_full_scores_val_set[0]}")
            if self.combined_validation_llm_judge:
                self.logger.log(
                    f"Iteration {gepa_state.i+1}: Base program retained-validation score for combined selection: "
                    f"{self.get_combined_validation_score_for_program(gepa_state, 0)}"
                )
        self.optimizer_phase_start_time = time.perf_counter()
        if not os.path.exists(os.path.join(self.run_dir, "metric_call_checkpoints.jsonl")):
            self.append_metric_call_checkpoint(
                gepa_state=gepa_state,
                phase="base_program_init",
                metric_calls_before=gepa_state.total_num_evals,
                metric_calls_after=gepa_state.total_num_evals,
                extra={"iteration": 0},
            )

        merges_due = 0
        total_merges_tested = 0

        last_iter_found_new_program = False

        merges_performed = ([], [])

        while (
            (self.num_iters is None or gepa_state.num_full_ds_evals < self.num_iters) and
            (self.max_evals_per_trainval_instance is None or gepa_state.total_num_evals_per_trainval_instance < self.max_evals_per_trainval_instance) and
            (self.max_metric_calls is None or gepa_state.total_num_evals < self.max_metric_calls) and
            (self.max_search_iterations is None or gepa_state.i + 1 < self.max_search_iterations) and
            (self.max_total_api_calls is None or self.get_total_search_api_calls() < self.max_total_api_calls) and
            (self.max_total_search_tokens is None or self.get_total_search_tokens() < self.max_total_search_tokens) and
            (self.api_call_hard_limit is None or self.get_total_search_api_calls() < self.api_call_hard_limit)
        ):
            assert gepa_state.is_consistent(), "GEPA state is inconsistent, please check the implementation"
            try:
                gepa_state.save(self.run_dir)
                gepa_state.i += 1
                gepa_state.full_program_trace.append({"i": gepa_state.i})
                gepa_state.full_program_trace[-1]["metric_calls_before_iteration"] = gepa_state.total_num_evals

                if merges_due > 0 and last_iter_found_new_program and self.use_merge:
                    last_iter_found_new_program = False
                    gepa_state.full_program_trace[-1]['invoked_merge'] = True

                    pareto_front_programs = self.get_pareto_front_programs(gepa_state)

                    merge_candidates = find_dominator_programs(pareto_front_programs, gepa_state.per_program_tracked_scores)
                    merge_output = sample_and_attempt_merge_programs_by_common_predictors(
                        agg_scores=gepa_state.per_program_tracked_scores,
                        rng=gepa_state.rng1,
                        merge_candidates=merge_candidates,
                        merges_performed=merges_performed,
                        program_candidates=gepa_state.program_candidates,
                        parent_program_for_candidate=gepa_state.parent_program_for_candidate,
                    )

                    if merge_output[0]:
                        gepa_state.full_program_trace[-1]['merged'] = True
                        success, new_program, id1, id2, ancestor = merge_output
                        assert success, "Merge output should be successful"
                        self.logger.log(f"Iteration {gepa_state.i+1}: Merged programs {id1} and {id2} via ancestor {ancestor}")
                        gepa_state.full_program_trace[-1]['merged_entities'] = (id1, id2, ancestor)

                        merges_performed[0].append((id1, id2, ancestor))

                        mini_devset = None

                        if self.set_for_merge_minibatch == 'train':
                            subsample_ids = self.select_eval_subsample_for_merged_program(
                                gepa_state.prog_candidate_train_subscores[id1],
                                gepa_state.prog_candidate_train_subscores[id2],
                                gepa_state.rng1,
                            )
                            mini_devset = [trainset[i] for i in subsample_ids]
                            id1_subsample_scores = [gepa_state.prog_candidate_train_subscores[id1][i] for i in subsample_ids]
                            id2_subsample_scores = [gepa_state.prog_candidate_train_subscores[id2][i] for i in subsample_ids]
                        elif self.set_for_merge_minibatch == 'val':
                            subsample_ids = self.select_eval_subsample_for_merged_program(
                                gepa_state.prog_candidate_val_subscores[id1],
                                gepa_state.prog_candidate_val_subscores[id2],
                                gepa_state.rng1,
                            )
                            mini_devset = [valset[i] for i in subsample_ids]
                            id1_subsample_scores = [gepa_state.prog_candidate_val_subscores[id1][i] for i in subsample_ids]
                            id2_subsample_scores = [gepa_state.prog_candidate_val_subscores[id2][i] for i in subsample_ids]
                        elif self.set_for_merge_minibatch == 'both':
                            subsample_ids = self.select_eval_subsample_for_merged_program(
                                gepa_state.prog_candidate_train_subscores[id1] + gepa_state.prog_candidate_val_subscores[id1],
                                gepa_state.prog_candidate_train_subscores[id2] + gepa_state.prog_candidate_val_subscores[id2],
                                gepa_state.rng1,
                            )
                            mini_devset = [trainset[i] if i < len(trainset) else valset[i - len(trainset)] for i in subsample_ids]
                            id1_subsample_scores = [gepa_state.prog_candidate_train_subscores[id1][i] if i < len(trainset) else gepa_state.prog_candidate_val_subscores[id1][i - len(trainset)] for i in subsample_ids]
                            id2_subsample_scores = [gepa_state.prog_candidate_train_subscores[id2][i] if i < len(trainset) else gepa_state.prog_candidate_val_subscores[id2][i - len(trainset)] for i in subsample_ids]
                        else:
                            self.logger.log(f"Iteration {gepa_state.i+1}: Unknown set for merge minibatch: {self.set_for_merge_minibatch}")
                            raise ValueError(f"Unknown set for merge minibatch: {self.set_for_merge_minibatch}. Should be 'train' or 'val'.")

                        gepa_state.full_program_trace[-1]['subsample_ids'] = subsample_ids

                        subsample_evaluator_args = {**trainset_evaluator.__dict__}
                        subsample_evaluator_args['devset'] = mini_devset
                        subsample_evaluator_args['return_outputs'] = True
                        subsample_evaluator_args['return_all_scores'] = True
                        subsample_evaluator_args['max_errors'] = len(subsample_ids) * 100
                        subsample_evaluator = LegacyEvaluate(**subsample_evaluator_args)

                        minibatch_usage_before = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm or new_program.get_lm())
                        with lm_trace_context(
                            "merge_minibatch_eval",
                            iteration=gepa_state.i + 1,
                            parent_program_idx=[id1, id2],
                            subsample_ids=subsample_ids,
                        ):
                            new_program_subsample_scores = subsample_evaluator(new_program)[2]
                        self.increment_minibatch_usage_from_lm(
                            gepa_state,
                            dspy.dsp.utils.settings.lm or new_program.get_lm(),
                            minibatch_usage_before,
                        )

                        id1_subsample_score = sum(id1_subsample_scores)
                        id2_subsample_score = sum(id2_subsample_scores)
                        new_subsample_score = sum(new_program_subsample_scores)

                        gepa_state.full_program_trace[-1]['id1_subsample_scores'] = id1_subsample_scores
                        gepa_state.full_program_trace[-1]['id2_subsample_scores'] = id2_subsample_scores
                        gepa_state.full_program_trace[-1]['new_program_subsample_scores'] = new_program_subsample_scores

                        gepa_state.total_num_evals_per_trainval_instance += len(subsample_ids) / self.train_val_size
                        gepa_state.total_num_evals += len(subsample_ids)

                        if new_subsample_score >= max(id1_subsample_score, id2_subsample_score):
                            self.logger.log(f"Iteration {gepa_state.i+1}: New program subsample score {new_subsample_score} for merged program is better than min of both parents {id1_subsample_score} and {id2_subsample_score}. proceeding with full eval")
                        else:
                            self.logger.log(f"Iteration {gepa_state.i+1}: New program subsample score {new_subsample_score} is worse than both parent programs {id1_subsample_score} and {id2_subsample_score}, skipping merge")
                            continue

                        merges_due -= 1
                        total_merges_tested += 1

                        new_program_idx, linear_pareto_front_idx = self.run_full_eval_add_new_program_to_gepa_tree(
                            new_program=new_program,
                            gepa_state=gepa_state,
                            trainset_evaluator=trainset_evaluator,
                            valset_evaluator=valset_evaluator,
                            parent_program_idx=[id1, id2]
                        )
                        continue
                    else:
                        self.logger.log(f"Iteration {gepa_state.i+1}: No merge candidates found")

                last_iter_found_new_program = False

                curr_prog_id = self.select_next_candidate_to_update(gepa_state)
                curr_prog = gepa_state.program_candidates[curr_prog_id]

                gepa_state.full_program_trace[-1]['selected_program_candidate'] = curr_prog_id

                predictor_to_update_id = gepa_state.named_predictor_id_to_update_next_for_program_candidate[curr_prog_id]
                gepa_state.full_program_trace[-1]['predictor_to_update_id'] = predictor_to_update_id
                gepa_state.named_predictor_id_to_update_next_for_program_candidate[curr_prog_id] = (predictor_to_update_id + 1) % len(gepa_state.list_of_named_predictors)
                predictor_name_to_update = gepa_state.list_of_named_predictors[predictor_to_update_id]
                gepa_state.full_program_trace[-1]['predictor_name_to_update'] = predictor_name_to_update
                if predictor_name_to_update not in self.named_predictor_to_feedback_fn_map:
                    self.logger.log(f"Iteration {gepa_state.i+1}: Predictor {predictor_name_to_update} not in feedback map, skipping")
                    continue

                self.logger.log(f"Iteration {gepa_state.i+1}: Selected program candidate {curr_prog_id} with base score: {gepa_state.per_program_tracked_scores[curr_prog_id]}")
                self.logger.log(f"Iteration {gepa_state.i+1}: Updating predictor {predictor_name_to_update}")

                if self.use_wandb:
                    wandb.log({
                        "iteration": gepa_state.i+1,
                        "selected_program_candidate": curr_prog_id,
                        "predictor_to_update_id": predictor_to_update_id,
                    }, step=gepa_state.i+1)

                feedback_func = self.named_predictor_to_feedback_fn_map[predictor_name_to_update]
                module = None
                for m in curr_prog.named_predictors():
                    if m[0] == predictor_name_to_update:
                        module = m[1]
                        break
                assert module is not None

                subsample_ids = self.select_training_sample_and_update_shuffled_trainset(trainset, gepa_state.i)
                gepa_state.full_program_trace[-1]['subsample_ids'] = subsample_ids

                minibatch_usage_before = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm or curr_prog.get_lm())
                with lm_trace_context(
                    "minibatch_feedback",
                    iteration=gepa_state.i + 1,
                    selected_program_candidate=curr_prog_id,
                    predictor_name=predictor_name_to_update,
                    subsample_ids=subsample_ids,
                ):
                    dataset_with_feedback, subsample_score, subsample_scores = capture_module_trace_with_feedback(
                        module,
                        curr_prog,
                        [trainset[i] for i in subsample_ids],
                        self.metric_fn,
                        self.logger,
                        gepa_state,
                        self.skip_perfect_score and not self.feedback_only_llm_judge,
                        self.perfect_score,
                        failure_score=self.failure_score,
                        format_failure_score=self.failure_score, # TODO: Get a proper value for this
                        feedback_func=feedback_func,
                        add_format_failure_as_feedback=self.add_format_failure_as_feedback,
                        num_threads=self.num_threads,
                    )
                self.increment_minibatch_usage_from_lm(
                    gepa_state,
                    dspy.dsp.utils.settings.lm or curr_prog.get_lm(),
                    minibatch_usage_before,
                )

                gepa_state.full_program_trace[-1]['subsample_scores'] = subsample_scores

                if dataset_with_feedback is None or subsample_score is None:
                    metric_calls_before = gepa_state.total_num_evals
                    gepa_state.total_num_evals_per_trainval_instance += len(subsample_ids) / self.train_val_size
                    gepa_state.total_num_evals += len(subsample_ids)
                    self.append_metric_call_checkpoint(
                        gepa_state=gepa_state,
                        phase="feedback_subsample_no_feedback",
                        metric_calls_before=metric_calls_before,
                        metric_calls_after=gepa_state.total_num_evals,
                    )
                    self.logger.log(f"Iteration {gepa_state.i+1}: No feedback samples, skipping")
                    continue

                if self.use_wandb:
                    wandb.log({
                        "subsample_score": subsample_score,
                    }, step=gepa_state.i+1)

                instruction_propose_module = ProposeNewInstructionModule(
                    base_program=module,
                    instruction_lm=self.teacher_lm or dspy.dsp.utils.settings.lm or curr_prog.get_lm(),
                    dataset_with_feedback=dataset_with_feedback,
                    knowledgebase_qe=self.knowledgebase_qe)
                if self.teacher_lm is not None:
                    instruction_propose_module.instruction_propose_module.set_lm(self.teacher_lm)
                try:
                    with lm_trace_context(
                        "instruction_proposal",
                        iteration=gepa_state.i + 1,
                        selected_program_candidate=curr_prog_id,
                        predictor_name=predictor_name_to_update,
                    ):
                        output = instruction_propose_module.compile()
                    with open(os.path.join(self.run_dir, "instruction_proposer_inpouts.jsonl"), 'a') as f:
                        f.write(json.dumps(output, default=lambda x: {**x}) + "\n")
                    new_instruction = output['new_instruction']
                    module_output = output['module_output']
                    kb_info = output['kb_info']
                except Exception as e:
                    self.logger.log(f"Iteration {gepa_state.i+1}: Exception during instruction proposal: {e}")
                    self.logger.log(traceback.format_exc())

                    continue
                self.logger.log(f"Iteration {gepa_state.i+1}: Info retrieved from knowledge base: {kb_info}")
                self.logger.log(f"Iteration {gepa_state.i+1}: Proposed new instruction: {new_instruction}")

                current_instruction = module.signature.instructions
                curr_prog_lm = curr_prog.get_lm()
                new_program = curr_prog.deepcopy()
                new_program.set_lm(curr_prog_lm)
                new_program.named_predictors()[predictor_to_update_id][1].signature = new_program.named_predictors()[predictor_to_update_id][1].signature.with_instructions(new_instruction)

                metric_calls_before = gepa_state.total_num_evals
                gepa_state.total_num_evals_per_trainval_instance += len(subsample_ids) / self.train_val_size
                gepa_state.total_num_evals += len(subsample_ids)
                self.append_metric_call_checkpoint(
                    gepa_state=gepa_state,
                    phase="feedback_subsample",
                    metric_calls_before=metric_calls_before,
                    metric_calls_after=gepa_state.total_num_evals,
                )

                if self.selection_mode == 'validation':
                    subsample_evaluator_args = {**trainset_evaluator.__dict__}
                    subsample_evaluator_args['devset'] = [trainset[i] for i in subsample_ids]
                    subsample_evaluator_args['return_outputs'] = True
                    subsample_evaluator_args['return_all_scores'] = True
                    subsample_evaluator_args['max_errors'] = len(subsample_ids) * 100
                    subsample_evaluator = LegacyEvaluate(**subsample_evaluator_args)
                    minibatch_usage_before = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm or new_program.get_lm())
                    with lm_trace_context(
                        "minibatch_candidate_eval",
                        iteration=gepa_state.i + 1,
                        selected_program_candidate=curr_prog_id,
                        predictor_name=predictor_name_to_update,
                        subsample_ids=subsample_ids,
                    ):
                        new_subsample_scores = subsample_evaluator(new_program)[2]
                    self.increment_minibatch_usage_from_lm(
                        gepa_state,
                        dspy.dsp.utils.settings.lm or new_program.get_lm(),
                        minibatch_usage_before,
                    )
                    new_subsample_score = sum(new_subsample_scores)

                    gepa_state.full_program_trace[-1]['new_subsample_scores'] = new_subsample_scores

                    metric_calls_before = gepa_state.total_num_evals
                    gepa_state.total_num_evals_per_trainval_instance += len(subsample_ids) / self.train_val_size
                    gepa_state.total_num_evals += len(subsample_ids)
                    self.append_metric_call_checkpoint(
                        gepa_state=gepa_state,
                        phase="candidate_subsample_eval",
                        metric_calls_before=metric_calls_before,
                        metric_calls_after=gepa_state.total_num_evals,
                        extra={"new_subsample_score": new_subsample_score},
                    )

                    self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score: {new_subsample_score}")
                    if self.use_wandb:
                        wandb.log({
                            "new_subsample_score": new_subsample_score,
                        }, step=gepa_state.i+1)

                    if new_subsample_score <= subsample_score:
                        if self.always_validate_for_teacher_memory:
                            validation_usage_before = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm or new_program.get_lm())
                            with lm_trace_context(
                                "validation_teacher_memory_eval",
                                iteration=gepa_state.i + 1,
                                selected_program_candidate=curr_prog_id,
                                predictor_name=predictor_name_to_update,
                            ):
                                new_valset_score, _, new_valset_subscores = valset_evaluator(new_program)
                            self.increment_validation_usage_from_lm(
                                gepa_state,
                                dspy.dsp.utils.settings.lm or new_program.get_lm(),
                                validation_usage_before,
                            )
                            metric_calls_before = gepa_state.total_num_evals
                            gepa_state.total_num_evals_per_trainval_instance += len(new_valset_subscores) / self.train_val_size
                            gepa_state.total_num_evals += len(new_valset_subscores)
                            old_valset_score = gepa_state.program_full_scores_val_set[curr_prog_id]
                            self.append_validation_teacher_memory_record_from_scores(
                                gepa_state=gepa_state,
                                selected_program_candidate=curr_prog_id,
                                predictor_name=predictor_name_to_update,
                                old_instruction=current_instruction,
                                new_instruction=new_instruction,
                                dataset_with_feedback=dataset_with_feedback,
                                old_subsample_score=subsample_score,
                                new_subsample_score=new_subsample_score,
                                old_validation_score=old_valset_score,
                                new_validation_score=new_valset_score,
                                record_suffix="teacher_memory_rejected",
                            )
                            self.append_metric_call_checkpoint(
                                gepa_state=gepa_state,
                                phase="teacher_memory_candidate_validation_eval",
                                metric_calls_before=metric_calls_before,
                                metric_calls_after=gepa_state.total_num_evals,
                                extra={
                                    "old_validation_score": old_valset_score,
                                    "new_validation_score": new_valset_score,
                                    "new_subsample_score": new_subsample_score,
                                    "teacher_memory_only": True,
                                },
                            )
                        self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score is not better, skipping")
                        continue

                    self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score is better, going from {subsample_score} to {new_subsample_score}, updating program candidate!")
                    last_iter_found_new_program = True
                    new_program_idx, linear_pareto_front_idx = self.run_full_eval_add_new_program_to_gepa_tree(
                        new_program=new_program,
                        gepa_state=gepa_state,
                        trainset_evaluator=trainset_evaluator,
                        valset_evaluator=valset_evaluator,
                        parent_program_idx=[curr_prog_id]
                    )
                    self.append_validation_teacher_memory_record(
                        gepa_state=gepa_state,
                        selected_program_candidate=curr_prog_id,
                        new_program_idx=new_program_idx,
                        predictor_name=predictor_name_to_update,
                        old_instruction=current_instruction,
                        new_instruction=new_instruction,
                        dataset_with_feedback=dataset_with_feedback,
                        old_subsample_score=subsample_score,
                        new_subsample_score=new_subsample_score,
                    )
                    if self.validation_sidecar_judge_alignment:
                        effective_alignment_judge_lm = self.judge_lm or self.teacher_lm or dspy.dsp.utils.settings.lm or curr_prog.get_lm()
                        if effective_alignment_judge_lm is None:
                            self.logger.log(
                                f"Iteration {gepa_state.i+1}: No sidecar judge LM available for warmup alignment logging."
                            )
                        else:
                            alignment_memory_records = self.load_alignment_memory_records()
                            retrieved_alignment_records = retrieve_relevant_memory_records(
                                alignment_memory_records,
                                predictor_name=predictor_name_to_update,
                                old_instruction=current_instruction,
                                new_instruction=new_instruction,
                                top_k=self.judge_memory_top_k,
                                same_predictor_only=self.judge_memory_same_predictor_only,
                            )
                            alignment_summary = summarize_alignment_memory_bank(
                                alignment_memory_records,
                                predictor_name=predictor_name_to_update,
                            ) if alignment_memory_records else None
                            alignment_context = format_alignment_memory_records(
                                retrieved_alignment_records,
                                max_cases=self.judge_memory_top_k,
                            ) if retrieved_alignment_records else None
                            with lm_trace_context(
                                "alignment_judge",
                                iteration=gepa_state.i + 1,
                                selected_program_candidate=curr_prog_id,
                                predictor_name=predictor_name_to_update,
                            ):
                                student_prediction = predict_alignment_pairwise_preference(
                                    judge_lm=effective_alignment_judge_lm,
                                    predictor_name=predictor_name_to_update,
                                    old_instruction=current_instruction,
                                    new_instruction=new_instruction,
                                    dataset_with_feedback=dataset_with_feedback,
                                    alignment_summary=alignment_summary,
                                    alignment_context=alignment_context,
                                )
                            alignment_record = self.append_validation_alignment_memory_record(
                                gepa_state=gepa_state,
                                selected_program_candidate=curr_prog_id,
                                new_program_idx=new_program_idx,
                                predictor_name=predictor_name_to_update,
                                old_instruction=current_instruction,
                                new_instruction=new_instruction,
                                dataset_with_feedback=dataset_with_feedback,
                                student_prediction=student_prediction,
                                retrieved_alignment_records=retrieved_alignment_records,
                            )
                            gepa_state.full_program_trace[-1]['warmup_alignment_probe'] = {
                                "student_preferred_prompt": alignment_record.get("student_preferred_prompt"),
                                "student_old_pairwise_score": alignment_record.get("student_old_pairwise_score"),
                                "student_new_pairwise_score": alignment_record.get("student_new_pairwise_score"),
                                "ranking_match": alignment_record.get("ranking_match"),
                                "teacher_preferred_prompt": alignment_record.get("teacher_preferred_prompt"),
                                "teacher_correction": alignment_record.get("teacher_correction"),
                            }
                            self.logger.log(
                                f"Iteration {gepa_state.i+1}: Warmup alignment probe "
                                f"student={alignment_record.get('student_preferred_prompt')} "
                                f"teacher={alignment_record.get('teacher_preferred_prompt')} "
                                f"match={alignment_record.get('ranking_match')}."
                            )
                else:
                    effective_judge_lm = self.judge_lm or self.teacher_lm or dspy.dsp.utils.settings.lm or curr_prog.get_lm()
                    if effective_judge_lm is None:
                        self.logger.log(f"Iteration {gepa_state.i+1}: No judge LM available, skipping candidate.")
                        continue

                    learned_judge_guide = self.load_learned_judge_guide()
                    judge_memory_records = []
                    retrieved_memory_records = []
                    memory_summary = None
                    memory_context = None
                    alignment_memory_records = []
                    retrieved_alignment_records = []
                    alignment_summary = None
                    alignment_context = None
                    if self.judge_strict_learned_guide:
                        self.logger.log(
                            f"Iteration {gepa_state.i+1}: Strict learned-guide judge enabled; skipping teacher/alignment memory retrieval."
                        )
                    else:
                        judge_memory_records = self.load_judge_memory_records()
                        retrieved_memory_records = retrieve_relevant_memory_records(
                            judge_memory_records,
                            predictor_name=predictor_name_to_update,
                            old_instruction=current_instruction,
                            new_instruction=new_instruction,
                            top_k=self.judge_memory_top_k,
                            same_predictor_only=self.judge_memory_same_predictor_only,
                        )
                        memory_summary = summarize_memory_bank(
                            judge_memory_records,
                            predictor_name=predictor_name_to_update,
                        ) if judge_memory_records else None
                        memory_context = format_memory_records(
                            retrieved_memory_records,
                            max_cases=self.judge_memory_top_k,
                        ) if retrieved_memory_records else None
                        alignment_memory_records = self.load_alignment_memory_records()
                        retrieved_alignment_records = retrieve_relevant_memory_records(
                            alignment_memory_records,
                            predictor_name=predictor_name_to_update,
                            old_instruction=current_instruction,
                            new_instruction=new_instruction,
                            top_k=self.judge_memory_top_k,
                            same_predictor_only=self.judge_memory_same_predictor_only,
                        )
                        alignment_summary = summarize_alignment_memory_bank(
                            alignment_memory_records,
                            predictor_name=predictor_name_to_update,
                        ) if alignment_memory_records else None
                        alignment_context = format_alignment_memory_records(
                            retrieved_alignment_records,
                            max_cases=self.judge_memory_top_k,
                        ) if retrieved_alignment_records else None
                        if retrieved_memory_records:
                            self.append_judge_memory_retrieval_record(
                                gepa_state=gepa_state,
                                predictor_name=predictor_name_to_update,
                                selected_program_candidate=curr_prog_id,
                                retrieved_records=retrieved_memory_records,
                                retrieved_alignment_records=retrieved_alignment_records,
                            )
                            self.logger.log(
                                f"Iteration {gepa_state.i+1}: Retrieved {len(retrieved_memory_records)} teacher-memory cases for predictor {predictor_name_to_update}."
                            )
                        elif retrieved_alignment_records:
                            self.append_judge_memory_retrieval_record(
                                gepa_state=gepa_state,
                                predictor_name=predictor_name_to_update,
                                selected_program_candidate=curr_prog_id,
                                retrieved_records=[],
                                retrieved_alignment_records=retrieved_alignment_records,
                            )
                        if retrieved_alignment_records:
                            self.logger.log(
                                f"Iteration {gepa_state.i+1}: Retrieved {len(retrieved_alignment_records)} alignment-memory cases for predictor {predictor_name_to_update}."
                            )

                    if self.combined_validation_llm_judge:
                        if valset_evaluator is None:
                            raise ValueError("validation_llm_judge_combined requires a validation evaluator")

                        old_validation_score = self.get_combined_validation_score_for_program(gepa_state, curr_prog_id)
                        if old_validation_score is None:
                            validation_usage_before = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm or curr_prog.get_lm())
                            with lm_trace_context(
                                "validation_parent_recovery_eval",
                                iteration=gepa_state.i + 1,
                                selected_program_candidate=curr_prog_id,
                                predictor_name=predictor_name_to_update,
                            ):
                                old_valset_score, _, old_valset_subscores = valset_evaluator(curr_prog)
                            self.increment_validation_usage_from_lm(
                                gepa_state,
                                dspy.dsp.utils.settings.lm or curr_prog.get_lm(),
                                validation_usage_before,
                            )
                            metric_calls_before = gepa_state.total_num_evals
                            gepa_state.total_num_evals_per_trainval_instance += len(old_valset_subscores) / self.train_val_size
                            gepa_state.total_num_evals += len(old_valset_subscores)
                            old_validation_score = old_valset_score
                            self.append_combined_validation_score_to_state(
                                gepa_state=gepa_state,
                                new_program_idx=curr_prog_id,
                                validation_score=old_valset_score,
                                validation_subscores=old_valset_subscores,
                            )
                            self.append_metric_call_checkpoint(
                                gepa_state=gepa_state,
                                phase="combined_parent_validation_eval_recovery",
                                metric_calls_before=metric_calls_before,
                                metric_calls_after=gepa_state.total_num_evals,
                                extra={
                                    "program_idx": curr_prog_id,
                                    "old_validation_score": old_validation_score,
                                },
                            )

                        validation_usage_before = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm or new_program.get_lm())
                        with lm_trace_context(
                            "validation_candidate_eval",
                            iteration=gepa_state.i + 1,
                            selected_program_candidate=curr_prog_id,
                            predictor_name=predictor_name_to_update,
                        ):
                            new_validation_score, _, new_validation_subscores = valset_evaluator(new_program)
                        self.increment_validation_usage_from_lm(
                            gepa_state,
                            dspy.dsp.utils.settings.lm or new_program.get_lm(),
                            validation_usage_before,
                        )
                        metric_calls_before = gepa_state.total_num_evals
                        gepa_state.total_num_evals_per_trainval_instance += len(new_validation_subscores) / self.train_val_size
                        gepa_state.total_num_evals += len(new_validation_subscores)
                        self.append_metric_call_checkpoint(
                            gepa_state=gepa_state,
                            phase="combined_candidate_validation_eval",
                            metric_calls_before=metric_calls_before,
                            metric_calls_after=gepa_state.total_num_evals,
                            extra={
                                "old_validation_score": old_validation_score,
                                "new_validation_score": new_validation_score,
                                "validation_delta": new_validation_score - old_validation_score,
                            },
                        )

                        with lm_trace_context(
                            "llm_judge",
                            iteration=gepa_state.i + 1,
                            selected_program_candidate=curr_prog_id,
                            predictor_name=predictor_name_to_update,
                            selection_variant="feedback_only",
                            combined_selection=True,
                        ):
                            judge_decision = judge_prompt_candidate(
                                judge_lm=effective_judge_lm,
                                predictor_name=predictor_name_to_update,
                                old_instruction=current_instruction,
                                new_instruction=new_instruction,
                                dataset_with_feedback=dataset_with_feedback,
                                memory_summary=memory_summary,
                                memory_context=memory_context,
                                alignment_summary=alignment_summary,
                                alignment_context=alignment_context,
                                learned_judge_guide=learned_judge_guide,
                                selection_variant="feedback_only",
                            )
                        combined_metadata = self.compute_combined_selection_metadata(
                            old_validation_score=old_validation_score,
                            new_validation_score=new_validation_score,
                            judge_decision=judge_decision,
                        )
                        judge_decision_record = {
                            "iteration": gepa_state.i + 1,
                            "total_metric_calls_at_judge": gepa_state.total_num_evals,
                            "selected_program_candidate": curr_prog_id,
                            "num_candidates_before_judge": len(gepa_state.program_candidates),
                            "best_program_idx_before_judge": idxmax(gepa_state.per_program_tracked_scores),
                            "best_tracked_score_before_judge": max(gepa_state.per_program_tracked_scores),
                            "predictor_name": predictor_name_to_update,
                            "parent_program_idx": curr_prog_id,
                            "feedback_example_count": len(dataset_with_feedback),
                            "retrieved_memory_record_ids": [record.get("memory_record_id") for record in retrieved_memory_records],
                            "num_retrieved_memory_cases": len(retrieved_memory_records),
                            "retrieved_alignment_record_ids": [
                                record.get("alignment_record_id") or record.get("memory_record_id")
                                for record in retrieved_alignment_records
                            ],
                            "num_retrieved_alignment_cases": len(retrieved_alignment_records),
                            "old_instruction": current_instruction,
                            "new_instruction": new_instruction,
                            **judge_decision,
                            **combined_metadata,
                        }
                        gepa_state.full_program_trace[-1]['judge_decision'] = judge_decision_record
                        gepa_state.full_program_trace[-1]['combined_selection'] = combined_metadata
                        self.append_jsonl_record("judge_decisions.jsonl", judge_decision_record)

                        if not combined_metadata["combined_accept"]:
                            self.append_combined_validation_memory_records(
                                gepa_state=gepa_state,
                                selected_program_candidate=curr_prog_id,
                                new_program_idx=None,
                                predictor_name=predictor_name_to_update,
                                old_instruction=current_instruction,
                                new_instruction=new_instruction,
                                dataset_with_feedback=dataset_with_feedback,
                                old_validation_score=old_validation_score,
                                new_validation_score=new_validation_score,
                                judge_decision=judge_decision,
                                combined_metadata=combined_metadata,
                                retrieved_alignment_records=retrieved_alignment_records,
                            )
                            self.append_metric_call_checkpoint(
                                gepa_state=gepa_state,
                                phase="combined_reject",
                                metric_calls_before=gepa_state.total_num_evals,
                                metric_calls_after=gepa_state.total_num_evals,
                                extra={
                                    "judge_preferred_prompt": judge_decision["preferred_prompt"],
                                    "judge_confidence": judge_decision["confidence"],
                                    "judge_signed_confidence": combined_metadata.get("judge_signed_confidence"),
                                    "judge_component": combined_metadata.get("judge_component"),
                                    "combined_delta": combined_metadata["combined_delta"],
                                    "validation_delta": combined_metadata["validation_delta"],
                                },
                            )
                            self.logger.log(
                                f"Iteration {gepa_state.i+1}: Combined selection kept old prompt. "
                                f"combined_delta={combined_metadata['combined_delta']}, "
                                f"validation_delta={combined_metadata['validation_delta']}, "
                                f"judge_preferred={judge_decision['preferred_prompt']}, "
                                f"judge_confidence={judge_decision['confidence']}"
                            )
                            continue

                        surrogate_gain = max(self.combined_min_surrogate_gain, combined_metadata["combined_delta"])
                        surrogate_score = gepa_state.per_program_tracked_scores[curr_prog_id] + surrogate_gain
                        selection_metadata = {
                            **judge_decision_record,
                            "surrogate_gain": surrogate_gain,
                            "surrogate_score": surrogate_score,
                        }
                        last_iter_found_new_program = True
                        self.logger.log(
                            f"Iteration {gepa_state.i+1}: Combined selection accepted new prompt with surrogate score {surrogate_score}. "
                            f"combined_delta={combined_metadata['combined_delta']}, "
                            f"validation_delta={combined_metadata['validation_delta']}, "
                            f"judge_preferred={judge_decision['preferred_prompt']}, "
                            f"judge_confidence={judge_decision['confidence']}"
                        )

                        new_program_idx, linear_pareto_front_idx = self.run_surrogate_selection_add_new_program_to_gepa_tree(
                            new_program=new_program,
                            gepa_state=gepa_state,
                            parent_program_idx=[curr_prog_id],
                            surrogate_score=surrogate_score,
                            selection_metadata=selection_metadata,
                            new_instruction=new_instruction,
                        )
                        self.append_combined_validation_score_to_state(
                            gepa_state=gepa_state,
                            new_program_idx=new_program_idx,
                            validation_score=new_validation_score,
                            validation_subscores=new_validation_subscores,
                        )
                        self.append_combined_validation_memory_records(
                            gepa_state=gepa_state,
                            selected_program_candidate=curr_prog_id,
                            new_program_idx=new_program_idx,
                            predictor_name=predictor_name_to_update,
                            old_instruction=current_instruction,
                            new_instruction=new_instruction,
                            dataset_with_feedback=dataset_with_feedback,
                            old_validation_score=old_validation_score,
                            new_validation_score=new_validation_score,
                            judge_decision=judge_decision,
                            combined_metadata=combined_metadata,
                            retrieved_alignment_records=retrieved_alignment_records,
                        )
                        self.append_metric_call_checkpoint(
                            gepa_state=gepa_state,
                            phase="combined_accept",
                            metric_calls_before=gepa_state.total_num_evals,
                            metric_calls_after=gepa_state.total_num_evals,
                            extra={
                                "new_program_idx": new_program_idx,
                                "judge_confidence": judge_decision["confidence"],
                                "judge_signed_confidence": combined_metadata.get("judge_signed_confidence"),
                                "judge_component": combined_metadata.get("judge_component"),
                                "combined_delta": combined_metadata["combined_delta"],
                                "validation_delta": combined_metadata["validation_delta"],
                            },
                        )
                        if self.use_merge and total_merges_tested < self.max_merge_invocations:
                            merges_due += 1
                        continue

                    new_subsample_score = None
                    if self.score_aware_llm_judge:
                        subsample_evaluator_args = {**trainset_evaluator.__dict__}
                        subsample_evaluator_args['devset'] = [trainset[i] for i in subsample_ids]
                        subsample_evaluator_args['return_outputs'] = True
                        subsample_evaluator_args['return_all_scores'] = True
                        subsample_evaluator_args['max_errors'] = len(subsample_ids) * 100
                        subsample_evaluator = LegacyEvaluate(**subsample_evaluator_args)
                        minibatch_usage_before = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm or new_program.get_lm())
                        with lm_trace_context(
                            "minibatch_candidate_eval",
                            iteration=gepa_state.i + 1,
                            selected_program_candidate=curr_prog_id,
                            predictor_name=predictor_name_to_update,
                            subsample_ids=subsample_ids,
                            score_aware_judge=True,
                        ):
                            new_subsample_scores = subsample_evaluator(new_program)[2]
                        self.increment_minibatch_usage_from_lm(
                            gepa_state,
                            dspy.dsp.utils.settings.lm or new_program.get_lm(),
                            minibatch_usage_before,
                        )
                        new_subsample_score = sum(new_subsample_scores)
                        gepa_state.full_program_trace[-1]['new_subsample_scores'] = new_subsample_scores
                        metric_calls_before = gepa_state.total_num_evals
                        gepa_state.total_num_evals_per_trainval_instance += len(subsample_ids) / self.train_val_size
                        gepa_state.total_num_evals += len(subsample_ids)
                        self.append_metric_call_checkpoint(
                            gepa_state=gepa_state,
                            phase="candidate_subsample_eval",
                            metric_calls_before=metric_calls_before,
                            metric_calls_after=gepa_state.total_num_evals,
                            extra={"new_subsample_score": new_subsample_score},
                        )
                        self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score: {new_subsample_score}")
                        if self.use_wandb:
                            wandb.log({
                                "new_subsample_score": new_subsample_score,
                            }, step=gepa_state.i+1)
                        if new_subsample_score <= subsample_score:
                            self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score is not better, skipping")
                            continue
                        self.logger.log(f"Iteration {gepa_state.i+1}: New subsample score is better, going from {subsample_score} to {new_subsample_score}, updating program candidate!")
                        with lm_trace_context(
                            "llm_judge",
                            iteration=gepa_state.i + 1,
                            selected_program_candidate=curr_prog_id,
                            predictor_name=predictor_name_to_update,
                            selection_variant="score_aware",
                        ):
                            judge_decision = judge_prompt_candidate(
                                judge_lm=effective_judge_lm,
                                predictor_name=predictor_name_to_update,
                                old_instruction=current_instruction,
                                new_instruction=new_instruction,
                                dataset_with_feedback=dataset_with_feedback,
                                memory_summary=memory_summary,
                                memory_context=memory_context,
                                alignment_summary=alignment_summary,
                                alignment_context=alignment_context,
                                learned_judge_guide=learned_judge_guide,
                                old_subsample_score=subsample_score,
                                new_subsample_score=new_subsample_score,
                                selection_variant="score_aware",
                            )
                    else:
                        with lm_trace_context(
                            "llm_judge",
                            iteration=gepa_state.i + 1,
                            selected_program_candidate=curr_prog_id,
                            predictor_name=predictor_name_to_update,
                            selection_variant="feedback_only",
                        ):
                            judge_decision = judge_prompt_candidate(
                                judge_lm=effective_judge_lm,
                                predictor_name=predictor_name_to_update,
                                old_instruction=current_instruction,
                                new_instruction=new_instruction,
                                dataset_with_feedback=dataset_with_feedback,
                                memory_summary=memory_summary,
                                memory_context=memory_context,
                                alignment_summary=alignment_summary,
                                alignment_context=alignment_context,
                                learned_judge_guide=learned_judge_guide,
                                selection_variant="feedback_only",
                            )
                    judge_decision_record = {
                        "iteration": gepa_state.i + 1,
                        "total_metric_calls_at_judge": gepa_state.total_num_evals,
                        "selected_program_candidate": curr_prog_id,
                        "num_candidates_before_judge": len(gepa_state.program_candidates),
                        "best_program_idx_before_judge": idxmax(gepa_state.per_program_tracked_scores),
                        "best_tracked_score_before_judge": max(gepa_state.per_program_tracked_scores),
                        "predictor_name": predictor_name_to_update,
                        "parent_program_idx": curr_prog_id,
                        "feedback_example_count": len(dataset_with_feedback),
                        "retrieved_memory_record_ids": [record.get("memory_record_id") for record in retrieved_memory_records],
                        "num_retrieved_memory_cases": len(retrieved_memory_records),
                        "retrieved_alignment_record_ids": [
                            record.get("alignment_record_id") or record.get("memory_record_id")
                            for record in retrieved_alignment_records
                        ],
                        "num_retrieved_alignment_cases": len(retrieved_alignment_records),
                        "old_instruction": current_instruction,
                        "new_instruction": new_instruction,
                        **judge_decision,
                    }
                    if self.score_aware_llm_judge:
                        judge_decision_record["old_subsample_score"] = subsample_score
                        judge_decision_record["new_subsample_score"] = new_subsample_score
                    gepa_state.full_program_trace[-1]['judge_decision'] = judge_decision_record
                    self.append_jsonl_record("judge_decisions.jsonl", judge_decision_record)

                    if judge_decision["preferred_prompt"] != "new":
                        self.append_metric_call_checkpoint(
                            gepa_state=gepa_state,
                            phase="judge_reject",
                            metric_calls_before=gepa_state.total_num_evals,
                            metric_calls_after=gepa_state.total_num_evals,
                            extra={
                                "judge_preferred_prompt": judge_decision["preferred_prompt"],
                                "judge_confidence": judge_decision["confidence"],
                            },
                        )
                        self.logger.log(f"Iteration {gepa_state.i+1}: Judge kept old prompt. Reason: {judge_decision['short_reason']}")
                        continue

                    surrogate_gain = max(0.01, judge_decision["confidence"])
                    surrogate_score = gepa_state.per_program_tracked_scores[curr_prog_id] + surrogate_gain
                    selection_metadata = {
                        **judge_decision_record,
                        "surrogate_gain": surrogate_gain,
                        "surrogate_score": surrogate_score,
                    }
                    last_iter_found_new_program = True
                    self.logger.log(
                        f"Iteration {gepa_state.i+1}: Judge accepted new prompt with surrogate score {surrogate_score}. "
                        f"Confidence={judge_decision['confidence']}"
                    )

                    new_program_idx, linear_pareto_front_idx = self.run_surrogate_selection_add_new_program_to_gepa_tree(
                        new_program=new_program,
                        gepa_state=gepa_state,
                        parent_program_idx=[curr_prog_id],
                        surrogate_score=surrogate_score,
                        selection_metadata=selection_metadata,
                        new_instruction=new_instruction,
                    )
                    self.append_metric_call_checkpoint(
                        gepa_state=gepa_state,
                        phase="judge_accept",
                        metric_calls_before=gepa_state.total_num_evals,
                        metric_calls_after=gepa_state.total_num_evals,
                        extra={
                            "new_program_idx": new_program_idx,
                            "judge_confidence": judge_decision["confidence"],
                        },
                    )

                if self.use_merge and total_merges_tested < self.max_merge_invocations:
                    merges_due += 1

            except Exception as e:
                self.logger.log(f"Iteration {gepa_state.i+1}: Exception during optimization: {e}")
                self.logger.log(traceback.format_exc())
                continue
            finally:
                if gepa_state.i >= 0 and gepa_state.full_program_trace:
                    curr_trace = gepa_state.full_program_trace[-1]
                    if curr_trace.get("i") == gepa_state.i:
                        judge_decision = curr_trace.get("judge_decision", {})
                        combined_selection = curr_trace.get("combined_selection", {})
                        warmup_alignment_probe = curr_trace.get("warmup_alignment_probe", {})
                        optimizer_usage = self.get_lm_usage_snapshot(dspy.dsp.utils.settings.lm)
                        judge_usage = self.get_lm_usage_snapshot(self.judge_lm)
                        iteration_summary = {
                            "iteration": gepa_state.i + 1,
                            "selection_mode": self.selection_mode,
                            "accepted_update": "new_program_idx" in curr_trace,
                            "metric_calls_before_iteration": curr_trace.get("metric_calls_before_iteration"),
                            "metric_calls_after_iteration": gepa_state.total_num_evals,
                            "selected_program_candidate": curr_trace.get("selected_program_candidate"),
                            "predictor_to_update_id": curr_trace.get("predictor_to_update_id"),
                            "predictor_name_to_update": curr_trace.get("predictor_name_to_update"),
                            "new_program_idx": curr_trace.get("new_program_idx"),
                            "num_candidates": len(gepa_state.program_candidates),
                            "total_metric_calls": gepa_state.total_num_evals,
                            "total_num_evals_per_trainval_instance": gepa_state.total_num_evals_per_trainval_instance,
                            "best_program_idx": idxmax(gepa_state.per_program_tracked_scores),
                            "best_tracked_score": max(gepa_state.per_program_tracked_scores),
                            "judge_preferred_prompt": judge_decision.get("preferred_prompt"),
                            "judge_confidence": judge_decision.get("confidence"),
                            "judge_parse_status": judge_decision.get("parse_status"),
                            "combined_score_mode": combined_selection.get("combined_score_mode"),
                            "combined_validation_delta": combined_selection.get("validation_delta"),
                            "combined_validation_component": combined_selection.get("validation_component"),
                            "combined_weighted_validation_component": combined_selection.get("weighted_validation_component"),
                            "combined_judge_signed_confidence": combined_selection.get("judge_signed_confidence"),
                            "combined_judge_component": combined_selection.get("judge_component"),
                            "combined_weighted_judge_component": combined_selection.get("weighted_judge_component"),
                            "combined_delta": combined_selection.get("combined_delta"),
                            "combined_accept": combined_selection.get("combined_accept"),
                            "combined_validation_teacher_preferred_prompt": combined_selection.get("validation_teacher_preferred_prompt"),
                            "warmup_student_preferred_prompt": warmup_alignment_probe.get("student_preferred_prompt"),
                            "warmup_alignment_ranking_match": warmup_alignment_probe.get("ranking_match"),
                            "warmup_teacher_preferred_prompt": warmup_alignment_probe.get("teacher_preferred_prompt"),
                            "selection_surrogate_score": curr_trace.get("selection_surrogate_score"),
                            "validation_input_tokens": gepa_state.validation_input_tokens,
                            "validation_output_tokens": gepa_state.validation_output_tokens,
                            "validation_api_calls": gepa_state.validation_api_calls,
                            "validation_tokens": gepa_state.validation_input_tokens + gepa_state.validation_output_tokens,
                            "optimizer_input_tokens": optimizer_usage["input_tokens"],
                            "optimizer_output_tokens": optimizer_usage["output_tokens"],
                            "optimizer_total_tokens": optimizer_usage["input_tokens"] + optimizer_usage["output_tokens"],
                            "optimizer_api_calls": optimizer_usage["api_calls"],
                            "judge_input_tokens": judge_usage["input_tokens"],
                            "judge_output_tokens": judge_usage["output_tokens"],
                            "judge_total_tokens": judge_usage["input_tokens"] + judge_usage["output_tokens"],
                            "judge_api_calls": judge_usage["api_calls"],
                            "optimizer_elapsed_seconds": time.perf_counter() - self.optimizer_phase_start_time if getattr(self, "optimizer_phase_start_time", None) is not None else None,
                        }
                        iteration_summary.update(self.build_cost_accounting_payload(optimizer_usage, judge_usage, gepa_state))
                        self.append_jsonl_record("iteration_summary.jsonl", iteration_summary)

        gepa_state.save(self.run_dir)

        return gepa_state
