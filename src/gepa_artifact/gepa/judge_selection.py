import difflib
import json
from typing import Any

import dspy
from .judge_memory import summarize_feedback_examples

try:
    from dspy.teleprompt.bootstrap_finetune import FailedPrediction
except ImportError:
    from dspy.teleprompt.bootstrap_trace import FailedPrediction


FEEDBACK_ONLY_JUDGE_PROMPT_VERSION = "prompt_selection_feedback_only_v1"
FEEDBACK_ONLY_WITH_MEMORY_JUDGE_PROMPT_VERSION = "prompt_selection_feedback_only_with_memory_v1"
FEEDBACK_ONLY_WITH_ALIGNMENT_MEMORY_JUDGE_PROMPT_VERSION = "prompt_selection_feedback_only_with_alignment_memory_v1"
FEEDBACK_ONLY_WITH_LEARNED_GUIDE_JUDGE_PROMPT_VERSION = "prompt_selection_feedback_only_with_learned_guide_v1"
SCORE_AWARE_JUDGE_PROMPT_VERSION = "prompt_selection_score_aware_v1"
COMBINED_SCORE_PAIR_JUDGE_PROMPT_VERSION = "prompt_selection_combined_score_pair_v1"
_REQUIRED_KEYS = ("preferred_prompt", "confidence", "short_reason", "risk_note")
_ALIGNMENT_REQUIRED_KEYS = _REQUIRED_KEYS + ("old_score", "new_score")


def _json_default(value: Any):
    try:
        return {**value}
    except Exception:
        return repr(value)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _format_mapping(mapping: Any, max_chars: int) -> str:
    if mapping is None:
        return "null"
    try:
        text = json.dumps(mapping, indent=2, default=_json_default, sort_keys=True)
    except Exception:
        text = repr(mapping)
    return _truncate_text(text, max_chars)

def summarize_instruction_diff(old_instruction: str, new_instruction: str, max_chars: int = 1200) -> str:
    old_lines = old_instruction.strip().splitlines()
    new_lines = new_instruction.strip().splitlines()
    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old",
            tofile="new",
            lineterm="",
        )
    )
    if not diff_lines:
        return "No instruction text diff."
    return _truncate_text("\n".join(diff_lines), max_chars=max_chars)


def _extract_first_json_object(raw_text: str) -> dict[str, Any] | None:
    start = raw_text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw_text)):
        char = raw_text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw_text[start : idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_decision(parsed: dict[str, Any]) -> dict[str, Any] | None:
    if any(key not in parsed for key in _REQUIRED_KEYS):
        return None

    preferred_prompt = str(parsed["preferred_prompt"]).strip().lower()
    if preferred_prompt not in {"old", "new"}:
        return None

    try:
        confidence = float(parsed["confidence"])
    except (TypeError, ValueError):
        return None

    confidence = min(1.0, max(0.0, confidence))
    short_reason = str(parsed["short_reason"]).strip()
    risk_note = str(parsed["risk_note"]).strip()
    if not short_reason:
        return None
    if not risk_note:
        risk_note = "none"

    return {
        "preferred_prompt": preferred_prompt,
        "confidence": confidence,
        "short_reason": _truncate_text(short_reason, 240),
        "risk_note": _truncate_text(risk_note, 200),
    }


def _normalize_alignment_prediction(parsed: dict[str, Any]) -> dict[str, Any] | None:
    if any(key not in parsed for key in _ALIGNMENT_REQUIRED_KEYS):
        return None

    normalized = _normalize_decision(parsed)
    if normalized is None:
        return None

    try:
        old_score = float(parsed["old_score"])
        new_score = float(parsed["new_score"])
    except (TypeError, ValueError):
        return None

    normalized.update(
        {
            "old_score": min(100.0, max(0.0, old_score)),
            "new_score": min(100.0, max(0.0, new_score)),
        }
    )
    return normalized


def _normalize_score_pair_decision(parsed: dict[str, Any]) -> dict[str, Any] | None:
    normalized = _normalize_alignment_prediction(parsed)
    if normalized is None:
        return None

    raw_preferred_prompt = normalized["preferred_prompt"]
    score_delta = normalized["new_score"] - normalized["old_score"]
    if score_delta > 1e-9:
        score_preferred_prompt = "new"
    elif score_delta < -1e-9:
        score_preferred_prompt = "old"
    else:
        score_preferred_prompt = raw_preferred_prompt

    normalized.update(
        {
            "raw_preferred_prompt": raw_preferred_prompt,
            "preferred_prompt": score_preferred_prompt,
            "score_preferred_prompt": score_preferred_prompt,
            "judge_score_delta": score_delta,
            "score_pair_consistent": raw_preferred_prompt == score_preferred_prompt,
        }
    )
    return normalized


def make_alignment_student_probe_prompt(
    predictor_name: str,
    old_instruction: str,
    new_instruction: str,
    reflective_evidence_summary: str,
    alignment_summary: str | None = None,
    alignment_context: str | None = None,
) -> str:
    alignment_block = ""
    if alignment_summary or alignment_context:
        alignment_block = f"""

Historical alignment / correction memory:
{alignment_summary or "No aggregate alignment summary is available."}

Retrieved alignment / correction cases:
{alignment_context or "No retrieved alignment cases are available."}

If those historical cases show that a previous judge ranking disagreed with validation, use them as weak correction signals.
Do not copy them mechanically; reason about the current case.
"""

    return f"""You are the warmup student judge for GEPA.

Validation is the teacher. Your job is to predict which instruction would generalize better before seeing validation scores.

Return exactly one JSON object with these keys:
{{
  "old_score": float between 0 and 100,
  "new_score": float between 0 and 100,
  "preferred_prompt": "old" or "new",
  "confidence": float between 0 and 1,
  "short_reason": "short explanation",
  "risk_note": "short caveat"
}}

Rules:
- Compare only this old-vs-new prompt pair.
- Predict relative generalization quality on an internal 0-100 scale.
- Higher score means more likely to generalize.
- Keep short_reason under 35 words.
- Keep risk_note under 20 words.
- Output JSON only.

Target predictor: {predictor_name}

OLD instruction:
{old_instruction}

NEW instruction:
{new_instruction}

Reflective evidence summary:
{reflective_evidence_summary}
{alignment_block}
"""


def make_prompt_selection_judge_prompt(
    predictor_name: str,
    old_instruction: str,
    new_instruction: str,
    reflective_evidence_summary: str,
    memory_summary: str | None = None,
    memory_context: str | None = None,
    alignment_summary: str | None = None,
    alignment_context: str | None = None,
    learned_judge_guide: str | None = None,
    old_subsample_score: float | None = None,
    new_subsample_score: float | None = None,
    selection_variant: str = "feedback_only",
) -> str:
    if selection_variant in {"feedback_only", "combined_score_pair"}:
        memory_chunks = []
        if learned_judge_guide:
            memory_chunks.append(f"""

Learned judge guide distilled from warmup validation:
{learned_judge_guide}

Use this guide as weak learned preference guidance. Do not treat it as current validation evidence.
"""
            )
        if memory_summary or memory_context:
            memory_chunks.append(f"""

Historical validation-teacher memory:
{memory_summary or "No aggregate teacher-memory summary is available."}

Retrieved teacher-memory cases:
{memory_context or "No retrieved teacher-memory cases are available."}

Use the historical teacher-memory cases only as weak guidance about which kinds of prompt edits tended to help or hurt earlier.
They are not current numeric evidence for this candidate.
"""
            )
        if alignment_summary or alignment_context:
            memory_chunks.append(f"""

Historical alignment / correction memory:
{alignment_summary or "No aggregate alignment summary is available."}

Retrieved alignment / correction cases:
{alignment_context or "No retrieved alignment-correction cases are available."}

If a historical case shows that a previous judge ranking disagreed with validation, treat that as weak corrective guidance.
Do not copy past decisions mechanically; reason about the current evidence.
"""
            )
        memory_block = "".join(memory_chunks)
        if selection_variant == "combined_score_pair":
            diff_summary = summarize_instruction_diff(old_instruction, new_instruction)
            return f"""You are a prompt-selection judge for GEPA.

You must independently score the OLD and NEW instructions for likely held-out generalization.
Validation scores will be combined separately by the optimizer; do not invent or infer hidden validation results.
Base your scores only on the reflective evidence summary, instruction diff, and weak historical memory below.

Return exactly one JSON object with these keys:
{{
  "old_score": float between 0 and 100,
  "new_score": float between 0 and 100,
  "preferred_prompt": "old" or "new",
  "confidence": float between 0 and 1,
  "short_reason": "short explanation",
  "risk_note": "short caveat"
}}

Rules:
- Judge only this single predictor update.
- Higher score means more likely to generalize.
- Set preferred_prompt to "new" if new_score is higher than old_score; otherwise set it to "old".
- The optimizer will use new_score - old_score as the LLM judge component.
- Do not mention formatting outside the JSON object.
- Keep short_reason under 35 words.
- Keep risk_note under 20 words.

Target predictor: {predictor_name}

OLD instruction:
{old_instruction}

NEW instruction:
{new_instruction}

Instruction diff summary:
{diff_summary}

Reflective evidence summary:
{reflective_evidence_summary}
{memory_block}
"""
        return f"""You are a prompt-selection judge for GEPA.

You must choose which instruction is more likely to generalize better for the target predictor.
Base your decision only on the reflective evidence summary below from a tiny minibatch.
No numeric scores are provided. Do not assume any score evidence.

Return exactly one JSON object with these keys:
{{
  "preferred_prompt": "old" or "new",
  "confidence": float between 0 and 1,
  "short_reason": "short explanation",
  "risk_note": "short caveat"
}}

Rules:
- Judge only this single predictor update.
- Use the reflective evidence to decide which instruction better addresses the observed issues while remaining likely to generalize.
- Do not mention formatting outside the JSON object.
- Keep short_reason under 35 words.
- Keep risk_note under 20 words.

Target predictor: {predictor_name}

OLD instruction:
{old_instruction}

NEW instruction:
{new_instruction}

Reflective evidence summary:
{reflective_evidence_summary}
{memory_block}
"""
    if old_subsample_score is None or new_subsample_score is None:
        raise ValueError("score_aware judge selection requires old/new minibatch scores")
    diff_summary = summarize_instruction_diff(old_instruction, new_instruction)
    return f"""You are a careful prompt-selection judge for GEPA.

You must choose which instruction is more likely to generalize better for the target predictor.
The evidence comes from only a tiny reflective minibatch, so it may overfit.
Prefer OLD unless NEW shows a clear likely generalization benefit.
Use the observed minibatch scores only as weak evidence. The minibatch is tiny and may overfit.

Return exactly one JSON object with these keys:
{{
  "preferred_prompt": "old" or "new",
  "confidence": float between 0 and 1,
  "short_reason": "short explanation",
  "risk_note": "short caveat"
}}

Rules:
- Judge only this single predictor update.
- Do not mention formatting outside the JSON object.
- Keep short_reason under 35 words.
- Keep risk_note under 20 words.

Target predictor: {predictor_name}

OLD instruction:
{old_instruction}

NEW instruction:
{new_instruction}

Observed minibatch scores:
- OLD score: {old_subsample_score}
- NEW score: {new_subsample_score}

Instruction diff summary:
{diff_summary}

Reflective evidence summary:
{reflective_evidence_summary}
"""


def judge_prompt_candidate(
    judge_lm: dspy.LM,
    predictor_name: str,
    old_instruction: str,
    new_instruction: str,
    dataset_with_feedback: list[dict[str, Any]],
    memory_summary: str | None = None,
    memory_context: str | None = None,
    alignment_summary: str | None = None,
    alignment_context: str | None = None,
    learned_judge_guide: str | None = None,
    old_subsample_score: float | None = None,
    new_subsample_score: float | None = None,
    selection_variant: str = "feedback_only",
    max_tokens: int = 600,
) -> dict[str, Any]:
    include_scores = selection_variant == "score_aware"
    prompt = make_prompt_selection_judge_prompt(
        predictor_name=predictor_name,
        old_instruction=old_instruction,
        new_instruction=new_instruction,
        reflective_evidence_summary=summarize_feedback_examples(
            dataset_with_feedback,
            include_scores=include_scores,
        ),
        memory_summary=memory_summary,
        memory_context=memory_context,
        alignment_summary=alignment_summary,
        alignment_context=alignment_context,
        learned_judge_guide=learned_judge_guide,
        old_subsample_score=old_subsample_score,
        new_subsample_score=new_subsample_score,
        selection_variant=selection_variant,
    )
    if selection_variant == "feedback_only":
        if learned_judge_guide:
            prompt_version = FEEDBACK_ONLY_WITH_LEARNED_GUIDE_JUDGE_PROMPT_VERSION
        elif alignment_summary or alignment_context:
            prompt_version = FEEDBACK_ONLY_WITH_ALIGNMENT_MEMORY_JUDGE_PROMPT_VERSION
        elif memory_summary or memory_context:
            prompt_version = FEEDBACK_ONLY_WITH_MEMORY_JUDGE_PROMPT_VERSION
        else:
            prompt_version = FEEDBACK_ONLY_JUDGE_PROMPT_VERSION
    elif selection_variant == "combined_score_pair":
        prompt_version = COMBINED_SCORE_PAIR_JUDGE_PROMPT_VERSION
    else:
        prompt_version = SCORE_AWARE_JUDGE_PROMPT_VERSION

    raw_response = judge_lm(prompt, max_tokens=max_tokens)[0].strip()
    parsed = _extract_first_json_object(raw_response)
    if selection_variant == "combined_score_pair":
        normalized = _normalize_score_pair_decision(parsed) if parsed is not None else None
    else:
        normalized = _normalize_decision(parsed) if parsed is not None else None

    if normalized is None:
        fallback_scores = {}
        if selection_variant == "combined_score_pair":
            fallback_scores = {
                "old_score": 50.0,
                "new_score": 50.0,
                "raw_preferred_prompt": "old",
                "score_preferred_prompt": "old",
                "judge_score_delta": 0.0,
                "score_pair_consistent": True,
            }
        return {
            "preferred_prompt": "old",
            "confidence": 0.0,
            "short_reason": "Judge output could not be parsed; keeping old prompt.",
            "risk_note": "judge_parse_failure",
            "parse_status": "fallback_keep_old",
            "raw_response": raw_response,
            "selection_variant": selection_variant,
            "prompt_version": prompt_version,
            "used_teacher_memory": selection_variant in {"feedback_only", "combined_score_pair"} and bool(memory_summary or memory_context),
            "used_alignment_memory": selection_variant in {"feedback_only", "combined_score_pair"} and bool(alignment_summary or alignment_context),
            "used_learned_judge_guide": selection_variant in {"feedback_only", "combined_score_pair"} and bool(learned_judge_guide),
            **fallback_scores,
        }

    return {
        **normalized,
        "parse_status": "ok",
        "raw_response": raw_response,
        "selection_variant": selection_variant,
        "prompt_version": prompt_version,
        "used_teacher_memory": selection_variant in {"feedback_only", "combined_score_pair"} and bool(memory_summary or memory_context),
        "used_alignment_memory": selection_variant in {"feedback_only", "combined_score_pair"} and bool(alignment_summary or alignment_context),
        "used_learned_judge_guide": selection_variant in {"feedback_only", "combined_score_pair"} and bool(learned_judge_guide),
    }


def predict_alignment_pairwise_preference(
    judge_lm: dspy.LM,
    predictor_name: str,
    old_instruction: str,
    new_instruction: str,
    dataset_with_feedback: list[dict[str, Any]],
    alignment_summary: str | None = None,
    alignment_context: str | None = None,
    max_tokens: int = 700,
) -> dict[str, Any]:
    prompt = make_alignment_student_probe_prompt(
        predictor_name=predictor_name,
        old_instruction=old_instruction,
        new_instruction=new_instruction,
        reflective_evidence_summary=summarize_feedback_examples(
            dataset_with_feedback,
            include_scores=False,
        ),
        alignment_summary=alignment_summary,
        alignment_context=alignment_context,
    )
    raw_response = judge_lm(prompt, max_tokens=max_tokens)[0].strip()
    parsed = _extract_first_json_object(raw_response)
    normalized = _normalize_alignment_prediction(parsed) if parsed is not None else None

    if normalized is None:
        return {
            "old_score": 50.0,
            "new_score": 50.0,
            "preferred_prompt": "old",
            "confidence": 0.0,
            "short_reason": "Warmup student output could not be parsed; defaulting to old.",
            "risk_note": "student_parse_failure",
            "parse_status": "fallback_keep_old",
            "raw_response": raw_response,
            "prompt_version": "warmup_alignment_student_probe_v1",
            "used_alignment_memory": bool(alignment_summary or alignment_context),
        }

    return {
        **normalized,
        "parse_status": "ok",
        "raw_response": raw_response,
        "prompt_version": "warmup_alignment_student_probe_v1",
        "used_alignment_memory": bool(alignment_summary or alignment_context),
    }

