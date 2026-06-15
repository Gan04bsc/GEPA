import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def truncate_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _json_default(value: Any):
    try:
        return {**value}
    except Exception:
        return repr(value)


def format_mapping(mapping: Any, max_chars: int) -> str:
    if mapping is None:
        return "null"
    try:
        text = json.dumps(mapping, indent=2, default=_json_default, sort_keys=True)
    except Exception:
        text = repr(mapping)
    return truncate_text(text, max_chars)


def summarize_feedback_examples(
    dataset_with_feedback: list[dict[str, Any]],
    max_examples: int = 3,
    max_chars_per_block: int = 800,
    include_scores: bool = False,
) -> str:
    lines = []
    for example_idx, sample in enumerate(dataset_with_feedback[:max_examples], start=1):
        lines.extend(
            [
                f"Example {example_idx}",
                "Inputs:",
                format_mapping(sample.get("inputs"), max_chars=max_chars_per_block),
                "Generated output:",
                format_mapping(sample.get("generated_output"), max_chars=max_chars_per_block),
                "Feedback:",
                truncate_text(str(sample.get("feedback", "")), max_chars=max_chars_per_block),
            ]
        )
        if include_scores and "score" in sample:
            lines.append(f"Trace score: {sample['score']}")
        lines.append("")
    return "\n".join(lines).strip()


def teacher_label_from_scores(old_score: float | None, new_score: float | None, eps: float = 1e-9) -> str:
    if old_score is None or new_score is None:
        return "unknown"
    delta = new_score - old_score
    if delta > eps:
        return "better"
    if delta < -eps:
        return "worse"
    return "same"


def teacher_preferred_prompt(old_score: float | None, new_score: float | None, eps: float = 1e-9) -> str:
    return "new" if teacher_label_from_scores(old_score, new_score, eps=eps) == "better" else "old"


def delta_bucket(delta: float | None, strong_threshold: float = 0.05, weak_threshold: float = 1e-9) -> str:
    if delta is None:
        return "unknown"
    if delta >= strong_threshold:
        return "strong_up"
    if delta > weak_threshold:
        return "small_up"
    if delta <= -strong_threshold:
        return "strong_down"
    if delta < -weak_threshold:
        return "small_down"
    return "flat"


def load_memory_bank(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    memory_path = Path(path)
    if not memory_path.exists():
        return []
    records = []
    with memory_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize_memory_bank(records: list[dict[str, Any]], predictor_name: str | None = None) -> str:
    scoped = [
        record for record in records
        if predictor_name is None or record.get("predictor_name") == predictor_name
    ]
    if not scoped:
        if predictor_name is None:
            return "No historical teacher-memory cases are available."
        return f"No historical teacher-memory cases are available for predictor `{predictor_name}`."

    label_counts: dict[str, int] = {}
    for record in scoped:
        label = str(record.get("teacher_label", "unknown"))
        label_counts[label] = label_counts.get(label, 0) + 1

    pieces = [
        f"Historical teacher-memory cases available: {len(scoped)}.",
        "Teacher-label distribution: " + ", ".join(f"{label}={count}" for label, count in sorted(label_counts.items())),
    ]
    if predictor_name is not None:
        pieces.append(f"Retrieved context should prioritize predictor `{predictor_name}` when possible.")
    return " ".join(pieces)


def summarize_alignment_memory_bank(records: list[dict[str, Any]], predictor_name: str | None = None) -> str:
    scoped = [
        record for record in records
        if predictor_name is None or record.get("predictor_name") == predictor_name
    ]
    if not scoped:
        if predictor_name is None:
            return "No historical alignment-correction cases are available."
        return f"No historical alignment-correction cases are available for predictor `{predictor_name}`."

    aligned = 0
    misaligned = 0
    correction_counts: dict[str, int] = {}
    for record in scoped:
        if record.get("ranking_match"):
            aligned += 1
        else:
            misaligned += 1
        correction = str(record.get("teacher_correction") or record.get("correction_direction") or "unspecified")
        correction_counts[correction] = correction_counts.get(correction, 0) + 1

    pieces = [
        f"Historical alignment-correction cases available: {len(scoped)}.",
        f"Aligned={aligned}, misaligned={misaligned}.",
    ]
    if correction_counts:
        top_corrections = sorted(correction_counts.items(), key=lambda item: item[1], reverse=True)[:3]
        pieces.append(
            "Common correction patterns: " + ", ".join(f"{label}={count}" for label, count in top_corrections)
        )
    if predictor_name is not None:
        pieces.append(f"Retrieved context should prioritize predictor `{predictor_name}` when possible.")
    return " ".join(pieces)


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _score_record(
    record: dict[str, Any],
    predictor_name: str,
    old_instruction: str,
    new_instruction: str,
) -> float:
    score = 0.0
    if record.get("predictor_name") == predictor_name:
        score += 5.0

    record_old = str(record.get("old_instruction", ""))
    record_new = str(record.get("new_instruction", ""))
    score += SequenceMatcher(None, record_old[:1200], old_instruction[:1200]).ratio()
    score += SequenceMatcher(None, record_new[:1200], new_instruction[:1200]).ratio()
    score += _jaccard(_tokenize(record_old) | _tokenize(record_new), _tokenize(old_instruction) | _tokenize(new_instruction))

    label = record.get("teacher_label")
    if label == "better":
        score += 0.1
    elif label == "worse":
        score += 0.05
    return score


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_text = f"{a.get('old_instruction', '')}\n{a.get('new_instruction', '')}"
    b_text = f"{b.get('old_instruction', '')}\n{b.get('new_instruction', '')}"
    return max(
        SequenceMatcher(None, a_text[:1200], b_text[:1200]).ratio(),
        _jaccard(_tokenize(a_text), _tokenize(b_text)),
    )


def select_distilled_teacher_memory_records(
    records: list[dict[str, Any]],
    predictor_name: str | None = None,
    max_cases: int = 5,
    similarity_threshold: float = 0.86,
) -> list[dict[str, Any]]:
    if max_cases <= 0 or not records:
        return []

    scoped = [
        record for record in records
        if predictor_name is None or record.get("predictor_name") == predictor_name
    ]
    if not scoped and predictor_name is not None:
        scoped = list(records)

    candidates = []
    for record in scoped:
        label = record.get("teacher_label")
        delta = _safe_float(record.get("teacher_delta"))
        if label in {None, "unknown", "same"} and (delta is None or abs(delta) <= 1e-9):
            continue
        strength = abs(delta) if delta is not None else 0.0
        candidates.append((strength, record))

    if not candidates:
        return []

    ranked = [record for _, record in sorted(candidates, key=lambda item: item[0], reverse=True)]
    selected: list[dict[str, Any]] = []
    label_counts = {"better": 0, "worse": 0}
    for record in ranked:
        label = str(record.get("teacher_label", "unknown"))
        if label in label_counts and label_counts[label] >= max(1, max_cases // 2 + 1):
            continue
        if any(_record_similarity(record, kept) >= similarity_threshold for kept in selected):
            continue
        selected.append(record)
        if label in label_counts:
            label_counts[label] += 1
        if len(selected) >= max_cases:
            return selected

    for record in ranked:
        if record in selected:
            continue
        selected.append(record)
        if len(selected) >= max_cases:
            break
    return selected


def build_learned_judge_guide(
    records: list[dict[str, Any]],
    predictor_name: str | None = None,
    max_cases: int = 5,
    max_instruction_chars: int = 360,
    max_feedback_chars: int = 420,
) -> dict[str, Any]:
    selected = select_distilled_teacher_memory_records(
        records,
        predictor_name=predictor_name,
        max_cases=max_cases,
    )
    label_counts: dict[str, int] = {}
    for record in records:
        label = str(record.get("teacher_label", "unknown"))
        label_counts[label] = label_counts.get(label, 0) + 1

    lines = [
        "Learned validation-teacher guide for GEPA prompt selection.",
        "This guide is distilled from warmup validation decisions. Use it as weak guidance, not as direct score evidence.",
        "Prefer prompt edits that fix concrete minibatch failures while preserving task constraints and avoiding over-specific rules.",
        "Reject prompt edits when validation teacher evidence showed regressions, brittle formatting changes, or narrow memorization.",
        f"Warmup record count: {len(records)}.",
        "Teacher-label distribution: " + ", ".join(f"{label}={count}" for label, count in sorted(label_counts.items())),
        "",
        "Distilled teacher cases:",
    ]
    case_payloads = []
    for idx, record in enumerate(selected, start=1):
        case_payload = {
            "memory_record_id": record.get("memory_record_id"),
            "iteration": record.get("iteration"),
            "predictor_name": record.get("predictor_name"),
            "teacher_label": record.get("teacher_label"),
            "teacher_preferred_prompt": record.get("teacher_preferred_prompt"),
            "teacher_delta": record.get("teacher_delta"),
            "teacher_delta_bucket": record.get("teacher_delta_bucket"),
        }
        case_payloads.append(case_payload)
        lines.extend(
            [
                f"Case {idx}",
                f"Predictor: {record.get('predictor_name', 'unknown')}",
                f"Validation teacher preferred: {record.get('teacher_preferred_prompt', 'unknown')}",
                f"Teacher label/delta: {record.get('teacher_label', 'unknown')} / {record.get('teacher_delta', 'unknown')}",
                "Old instruction:",
                truncate_text(str(record.get("old_instruction", "")), max_instruction_chars),
                "New instruction:",
                truncate_text(str(record.get("new_instruction", "")), max_instruction_chars),
                "Reflective evidence:",
                truncate_text(str(record.get("reflective_evidence_summary", "")), max_feedback_chars),
                "",
            ]
        )

    return {
        "record_type": "learned_judge_guide_v1",
        "predictor_name": predictor_name,
        "warmup_record_count": len(records),
        "selected_case_count": len(selected),
        "label_counts": label_counts,
        "selected_cases": case_payloads,
        "guide_text": "\n".join(lines).strip(),
    }


def select_high_confidence_misalignment_records(
    records: list[dict[str, Any]],
    max_cases: int = 2,
) -> list[dict[str, Any]]:
    if max_cases <= 0 or not records:
        return []

    candidates = []
    for record in records:
        if record.get("ranking_match") is True:
            continue
        confidence = _safe_float(record.get("student_confidence"))
        if confidence is None:
            continue
        old_score = _safe_float(record.get("student_old_pairwise_score")) or 0.0
        new_score = _safe_float(record.get("student_new_pairwise_score")) or 0.0
        teacher_delta = _safe_float(record.get("teacher_delta")) or 0.0
        candidates.append((confidence, abs(new_score - old_score), abs(teacher_delta), record))

    ranked = sorted(candidates, key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [record for _, _, _, record in ranked[:max_cases]]


def build_learned_judge_guide_with_alignment(
    teacher_records: list[dict[str, Any]],
    alignment_records: list[dict[str, Any]],
    predictor_name: str | None = None,
    teacher_cases: int = 3,
    alignment_cases: int = 2,
    max_instruction_chars: int = 360,
    max_feedback_chars: int = 420,
) -> dict[str, Any]:
    selected_teacher = select_distilled_teacher_memory_records(
        teacher_records,
        predictor_name=predictor_name,
        max_cases=teacher_cases,
    )
    selected_alignment = select_high_confidence_misalignment_records(
        alignment_records,
        max_cases=alignment_cases,
    )

    label_counts: dict[str, int] = {}
    for record in teacher_records:
        label = str(record.get("teacher_label", "unknown"))
        label_counts[label] = label_counts.get(label, 0) + 1

    misalignment_count = sum(1 for record in alignment_records if record.get("ranking_match") is not True)
    lines = [
        "Learned validation-teacher guide for GEPA prompt selection.",
        "This guide is distilled from warmup validation decisions and high-confidence student-judge mistakes.",
        "Use it as weak guidance, not as direct score evidence.",
        "Prefer prompt edits that fix concrete minibatch failures while preserving task constraints and avoiding over-specific rules.",
        "Reject prompt edits when validation teacher evidence showed regressions, brittle formatting changes, narrow memorization, or high-confidence judge mistakes.",
        f"Warmup teacher record count: {len(teacher_records)}.",
        "Teacher-label distribution: " + ", ".join(f"{label}={count}" for label, count in sorted(label_counts.items())),
        f"Warmup alignment record count: {len(alignment_records)}; misaligned={misalignment_count}.",
        "",
        "Distilled validation-teacher cases:",
    ]

    teacher_payloads = []
    for idx, record in enumerate(selected_teacher, start=1):
        teacher_payloads.append(
            {
                "memory_record_id": record.get("memory_record_id"),
                "iteration": record.get("iteration"),
                "predictor_name": record.get("predictor_name"),
                "teacher_label": record.get("teacher_label"),
                "teacher_preferred_prompt": record.get("teacher_preferred_prompt"),
                "teacher_delta": record.get("teacher_delta"),
                "teacher_delta_bucket": record.get("teacher_delta_bucket"),
            }
        )
        lines.extend(
            [
                f"Teacher Case {idx}",
                f"Predictor: {record.get('predictor_name', 'unknown')}",
                f"Validation teacher preferred: {record.get('teacher_preferred_prompt', 'unknown')}",
                f"Teacher label/delta: {record.get('teacher_label', 'unknown')} / {record.get('teacher_delta', 'unknown')}",
                "Old instruction:",
                truncate_text(str(record.get("old_instruction", "")), max_instruction_chars),
                "New instruction:",
                truncate_text(str(record.get("new_instruction", "")), max_instruction_chars),
                "Reflective evidence:",
                truncate_text(str(record.get("reflective_evidence_summary", "")), max_feedback_chars),
                "",
            ]
        )

    lines.append("High-confidence judge-mistake cases:")
    alignment_payloads = []
    for idx, record in enumerate(selected_alignment, start=1):
        alignment_payloads.append(
            {
                "alignment_record_id": record.get("alignment_record_id"),
                "iteration": record.get("iteration"),
                "predictor_name": record.get("predictor_name"),
                "teacher_preferred_prompt": record.get("teacher_preferred_prompt"),
                "student_preferred_prompt": record.get("student_preferred_prompt"),
                "student_confidence": record.get("student_confidence"),
                "student_old_pairwise_score": record.get("student_old_pairwise_score"),
                "student_new_pairwise_score": record.get("student_new_pairwise_score"),
                "teacher_delta": record.get("teacher_delta"),
                "correction_direction": record.get("correction_direction"),
            }
        )
        lines.extend(
            [
                f"Judge Mistake Case {idx}",
                f"Predictor: {record.get('predictor_name', 'unknown')}",
                f"Student preferred: {record.get('student_preferred_prompt', 'unknown')} with confidence {record.get('student_confidence', 'unknown')}",
                f"Validation teacher preferred: {record.get('teacher_preferred_prompt', 'unknown')}",
                f"Student old/new scores: {record.get('student_old_pairwise_score', 'unknown')} / {record.get('student_new_pairwise_score', 'unknown')}",
                f"Teacher delta: {record.get('teacher_delta', 'unknown')}",
                "Teacher correction:",
                truncate_text(str(record.get("teacher_correction") or record.get("correction_direction") or ""), 240),
                "Old instruction:",
                truncate_text(str(record.get("old_instruction", "")), max_instruction_chars),
                "New instruction:",
                truncate_text(str(record.get("new_instruction", "")), max_instruction_chars),
                "Reflective evidence:",
                truncate_text(str(record.get("reflective_evidence_summary", "")), max_feedback_chars),
                "",
            ]
        )

    return {
        "record_type": "learned_judge_guide_v2_with_alignment",
        "predictor_name": predictor_name,
        "warmup_record_count": len(teacher_records),
        "alignment_record_count": len(alignment_records),
        "misalignment_record_count": misalignment_count,
        "selected_case_count": len(selected_teacher) + len(selected_alignment),
        "selected_teacher_case_count": len(selected_teacher),
        "selected_alignment_case_count": len(selected_alignment),
        "label_counts": label_counts,
        "selected_cases": teacher_payloads + alignment_payloads,
        "selected_teacher_cases": teacher_payloads,
        "selected_alignment_cases": alignment_payloads,
        "guide_text": "\n".join(lines).strip(),
    }


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


def _rules_source_record_summary(
    records: list[dict[str, Any]],
    max_records: int = 50,
    max_instruction_chars: int = 420,
    max_feedback_chars: int = 520,
) -> str:
    if not records:
        return "No warmup teacher records are available."

    sorted_records = sorted(records, key=lambda record: record.get("iteration") or 0)
    lines = []
    for idx, record in enumerate(sorted_records[:max_records], start=1):
        lines.extend(
            [
                f"Warmup Case {idx}",
                f"Record id: {record.get('memory_record_id', 'unknown')}",
                f"Iteration: {record.get('iteration', 'unknown')}",
                f"Predictor: {record.get('predictor_name', 'unknown')}",
                f"Validation teacher preferred: {record.get('teacher_preferred_prompt', 'unknown')}",
                f"Teacher label/delta: {record.get('teacher_label', 'unknown')} / {record.get('teacher_delta', 'unknown')}",
                f"Teacher old/new score: {record.get('teacher_old_val_score', 'unknown')} / {record.get('teacher_new_val_score', 'unknown')}",
                "Old instruction:",
                truncate_text(str(record.get("old_instruction", "")), max_instruction_chars),
                "New instruction:",
                truncate_text(str(record.get("new_instruction", "")), max_instruction_chars),
                "Feedback evidence:",
                truncate_text(str(record.get("reflective_evidence_summary", "")), max_feedback_chars),
                "",
            ]
        )
    return "\n".join(lines).strip()


def make_rules_library_prompt(
    teacher_records: list[dict[str, Any]],
    max_rules: int = 50,
    max_source_records: int = 50,
) -> str:
    source_records = _rules_source_record_summary(
        teacher_records,
        max_records=max_source_records,
    )
    return f"""You are distilling a GEPA prompt-selection rules library from warmup validation evidence.

The warmup cases below include prompt edits that helped, hurt, or were neutral according to validation-teacher scores.
Your task is to infer concrete rules that a later LLM judge can use when choosing between OLD and NEW prompts.

Return exactly one JSON object with this schema:
{{
  "summary": "brief summary of the warmup evidence",
  "rules": [
    {{
      "rule_id": "R1",
      "rule": "clear actionable rule",
      "applies_when": "specific condition where this rule applies",
      "avoid_when": "specific condition where this rule should not be applied",
      "evidence": "short evidence from warmup cases",
      "source_record_ids": ["record id"]
    }}
  ]
}}

Requirements:
- Produce at most {max_rules} rules. You may produce fewer rules if the evidence only supports fewer.
- Rules must be concrete and operational. Avoid vague advice like "be clear", "be accurate", or "use better reasoning" unless it is tied to a specific observable failure mode.
- Use both successful and failed edits. A rule can say when to prefer a new prompt or when to reject it.
- Do not copy the full warmup examples into the rules.
- Keep each rule under 40 words.
- Return JSON only.

Warmup validation-teacher cases:
{source_records}
"""


def parse_rules_library_response(
    raw_response: str,
    max_rules: int = 50,
) -> dict[str, Any]:
    parsed = _extract_first_json_object(raw_response)
    rules: list[dict[str, Any]] = []
    summary = ""
    parse_status = "ok"

    if parsed is None:
        parsed = {}
        parse_status = "parse_failure"

    if isinstance(parsed.get("summary"), str):
        summary = parsed["summary"].strip()

    raw_rules = parsed.get("rules", [])
    if not isinstance(raw_rules, list):
        raw_rules = []
        parse_status = "parse_failure"

    for idx, raw_rule in enumerate(raw_rules[:max_rules], start=1):
        if isinstance(raw_rule, str):
            rule = raw_rule.strip()
            payload = {
                "rule_id": f"R{idx}",
                "rule": rule,
                "applies_when": "unspecified",
                "avoid_when": "unspecified",
                "evidence": "unspecified",
                "source_record_ids": [],
            }
        elif isinstance(raw_rule, dict):
            rule = str(raw_rule.get("rule", "")).strip()
            payload = {
                "rule_id": str(raw_rule.get("rule_id") or f"R{idx}").strip(),
                "rule": rule,
                "applies_when": str(raw_rule.get("applies_when") or "unspecified").strip(),
                "avoid_when": str(raw_rule.get("avoid_when") or "unspecified").strip(),
                "evidence": str(raw_rule.get("evidence") or "unspecified").strip(),
                "source_record_ids": raw_rule.get("source_record_ids") or [],
            }
        else:
            continue

        if not payload["rule"]:
            continue
        if not isinstance(payload["source_record_ids"], list):
            payload["source_record_ids"] = [str(payload["source_record_ids"])]
        payload["source_record_ids"] = [str(item) for item in payload["source_record_ids"]]
        rules.append(payload)

    return {
        "record_type": "warmup_rules_library_v1",
        "parse_status": parse_status,
        "summary": summary,
        "rule_count": len(rules),
        "max_rules": max_rules,
        "rules": rules,
        "raw_response": raw_response,
    }


def format_rules_library(rules_payload: dict[str, Any]) -> str:
    rules = rules_payload.get("rules") or []
    if not rules:
        return (
            "No parsed warmup rules are available. "
            "Fall back to the current reflective evidence and be conservative about accepting prompt edits."
        )

    lines = [
        "Warmup rules library for GEPA prompt selection.",
        "Use these rules as weak learned policy, not as current validation evidence.",
    ]
    if rules_payload.get("summary"):
        lines.extend(["Evidence summary:", truncate_text(str(rules_payload["summary"]), 900)])
    lines.append("")
    lines.append("Rules:")
    for rule in rules:
        lines.extend(
            [
                f"{rule.get('rule_id', 'R?')}. {rule.get('rule', '')}",
                f"Applies when: {rule.get('applies_when', 'unspecified')}",
                f"Avoid when: {rule.get('avoid_when', 'unspecified')}",
                f"Evidence: {truncate_text(str(rule.get('evidence', 'unspecified')), 260)}",
                f"Source records: {', '.join(rule.get('source_record_ids') or []) or 'unspecified'}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def build_rules_augmented_judge_guide(
    rules_payload: dict[str, Any],
    teacher_records: list[dict[str, Any]],
    alignment_records: list[dict[str, Any]],
    *,
    include_fewshot: bool,
    teacher_cases: int = 3,
    alignment_cases: int = 2,
    max_instruction_chars: int = 360,
    max_feedback_chars: int = 420,
) -> dict[str, Any]:
    lines = [
        "Learned rules guide for GEPA prompt selection.",
        "This guide is distilled after warmup validation. Use it as weak guidance together with the current minibatch feedback.",
        "",
        format_rules_library(rules_payload),
    ]

    selected_teacher: list[dict[str, Any]] = []
    selected_alignment: list[dict[str, Any]] = []

    if include_fewshot:
        selected_teacher = select_distilled_teacher_memory_records(
            teacher_records,
            max_cases=teacher_cases,
        )
        selected_alignment = select_high_confidence_misalignment_records(
            alignment_records,
            max_cases=alignment_cases,
        )
        lines.extend(
            [
                "",
                "Few-shot validation-teacher cases for calibration.",
                "Use these examples only to interpret the rules; do not retrieve or infer any other memory.",
                "",
                "Top validation-teacher pairs:",
            ]
        )
        for idx, record in enumerate(selected_teacher, start=1):
            lines.extend(
                [
                    f"Teacher Pair {idx}",
                    f"Predictor: {record.get('predictor_name', 'unknown')}",
                    f"Validation teacher preferred: {record.get('teacher_preferred_prompt', 'unknown')}",
                    f"Teacher label/delta: {record.get('teacher_label', 'unknown')} / {record.get('teacher_delta', 'unknown')}",
                    "Old instruction:",
                    truncate_text(str(record.get("old_instruction", "")), max_instruction_chars),
                    "New instruction:",
                    truncate_text(str(record.get("new_instruction", "")), max_instruction_chars),
                    "Reflective evidence:",
                    truncate_text(str(record.get("reflective_evidence_summary", "")), max_feedback_chars),
                    "",
                ]
            )

        lines.append("High-confidence judge-mistake pairs:")
        for idx, record in enumerate(selected_alignment, start=1):
            lines.extend(
                [
                    f"Alignment Pair {idx}",
                    f"Predictor: {record.get('predictor_name', 'unknown')}",
                    f"Student preferred: {record.get('student_preferred_prompt', 'unknown')} with confidence {record.get('student_confidence', 'unknown')}",
                    f"Validation teacher preferred: {record.get('teacher_preferred_prompt', 'unknown')}",
                    f"Student old/new scores: {record.get('student_old_pairwise_score', 'unknown')} / {record.get('student_new_pairwise_score', 'unknown')}",
                    f"Teacher delta: {record.get('teacher_delta', 'unknown')}",
                    "Teacher correction:",
                    truncate_text(str(record.get("teacher_correction") or record.get("correction_direction") or ""), 240),
                    "Old instruction:",
                    truncate_text(str(record.get("old_instruction", "")), max_instruction_chars),
                    "New instruction:",
                    truncate_text(str(record.get("new_instruction", "")), max_instruction_chars),
                    "Reflective evidence:",
                    truncate_text(str(record.get("reflective_evidence_summary", "")), max_feedback_chars),
                    "",
                ]
            )

    return {
        "record_type": "rules_augmented_judge_guide_v1",
        "include_fewshot": include_fewshot,
        "warmup_record_count": len(teacher_records),
        "alignment_record_count": len(alignment_records),
        "rule_count": rules_payload.get("rule_count", 0),
        "selected_case_count": len(selected_teacher) + len(selected_alignment),
        "selected_teacher_case_count": len(selected_teacher),
        "selected_alignment_case_count": len(selected_alignment),
        "rules": rules_payload.get("rules", []),
        "selected_teacher_cases": [
            {
                "memory_record_id": record.get("memory_record_id"),
                "iteration": record.get("iteration"),
                "predictor_name": record.get("predictor_name"),
                "teacher_label": record.get("teacher_label"),
                "teacher_preferred_prompt": record.get("teacher_preferred_prompt"),
                "teacher_delta": record.get("teacher_delta"),
            }
            for record in selected_teacher
        ],
        "selected_alignment_cases": [
            {
                "alignment_record_id": record.get("alignment_record_id"),
                "iteration": record.get("iteration"),
                "predictor_name": record.get("predictor_name"),
                "teacher_preferred_prompt": record.get("teacher_preferred_prompt"),
                "student_preferred_prompt": record.get("student_preferred_prompt"),
                "student_confidence": record.get("student_confidence"),
                "teacher_delta": record.get("teacher_delta"),
                "correction_direction": record.get("correction_direction"),
            }
            for record in selected_alignment
        ],
        "guide_text": "\n".join(lines).strip(),
    }


def retrieve_relevant_memory_records(
    records: list[dict[str, Any]],
    predictor_name: str,
    old_instruction: str,
    new_instruction: str,
    top_k: int = 3,
    same_predictor_only: bool = True,
) -> list[dict[str, Any]]:
    if top_k <= 0 or not records:
        return []

    scoped = records
    if same_predictor_only:
        same_predictor_records = [record for record in records if record.get("predictor_name") == predictor_name]
        if same_predictor_records:
            scoped = same_predictor_records

    ranked = sorted(
        scoped,
        key=lambda record: _score_record(record, predictor_name, old_instruction, new_instruction),
        reverse=True,
    )
    return ranked[:top_k]


def format_memory_records(
    records: list[dict[str, Any]],
    max_cases: int = 3,
    max_instruction_chars: int = 500,
    max_feedback_chars: int = 800,
) -> str:
    if not records:
        return "No retrieved teacher-memory cases."

    lines = []
    for idx, record in enumerate(records[:max_cases], start=1):
        lines.extend(
            [
                f"Memory Case {idx}",
                f"Predictor: {record.get('predictor_name', 'unknown')}",
                f"Teacher label: {record.get('teacher_label', 'unknown')}",
                f"Teacher preferred prompt: {record.get('teacher_preferred_prompt', 'unknown')}",
                f"Teacher delta bucket: {record.get('teacher_delta_bucket', 'unknown')}",
                "Old instruction:",
                truncate_text(str(record.get("old_instruction", "")), max_instruction_chars),
                "New instruction:",
                truncate_text(str(record.get("new_instruction", "")), max_instruction_chars),
                "Reflective evidence summary:",
                truncate_text(str(record.get("reflective_evidence_summary", "")), max_feedback_chars),
                "",
            ]
        )
    return "\n".join(lines).strip()


def format_alignment_memory_records(
    records: list[dict[str, Any]],
    max_cases: int = 3,
    max_instruction_chars: int = 500,
    max_feedback_chars: int = 800,
) -> str:
    if not records:
        return "No retrieved alignment-correction cases."

    lines = []
    for idx, record in enumerate(records[:max_cases], start=1):
        lines.extend(
            [
                f"Alignment Case {idx}",
                f"Predictor: {record.get('predictor_name', 'unknown')}",
                f"Teacher preferred prompt: {record.get('teacher_preferred_prompt', 'unknown')}",
                f"Student preferred prompt: {record.get('student_preferred_prompt', 'unknown')}",
                f"Ranking match: {record.get('ranking_match', 'unknown')}",
                f"Teacher delta bucket: {record.get('teacher_delta_bucket', 'unknown')}",
                f"Student old/new scores: {record.get('student_old_pairwise_score', 'unknown')} / {record.get('student_new_pairwise_score', 'unknown')}",
                "Teacher correction:",
                truncate_text(
                    str(record.get("teacher_correction") or record.get("correction_direction") or ""),
                    240,
                ),
                "Old instruction:",
                truncate_text(str(record.get("old_instruction", "")), max_instruction_chars),
                "New instruction:",
                truncate_text(str(record.get("new_instruction", "")), max_instruction_chars),
                "Reflective evidence summary:",
                truncate_text(str(record.get("reflective_evidence_summary", "")), max_feedback_chars),
                "",
            ]
        )
    return "\n".join(lines).strip()
