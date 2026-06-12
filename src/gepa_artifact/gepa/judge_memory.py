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
