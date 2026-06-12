from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherPair:
    predictor_name: str
    old_prompt: str
    new_prompt: str
    feedback: str
    old_validation_score: float | None
    new_validation_score: float | None

    @property
    def validation_delta(self) -> float | None:
        if self.old_validation_score is None or self.new_validation_score is None:
            return None
        return self.new_validation_score - self.old_validation_score

    @property
    def preferred_prompt(self) -> str:
        delta = self.validation_delta
        return "new" if delta is not None and delta > 0 else "old"


def load_teacher_pairs(path: str | Path) -> list[TeacherPair]:
    records: list[TeacherPair] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                TeacherPair(
                    predictor_name=str(payload.get("predictor_name", "")),
                    old_prompt=str(payload.get("old_instruction") or payload.get("old_prompt") or ""),
                    new_prompt=str(payload.get("new_instruction") or payload.get("new_prompt") or ""),
                    feedback=str(payload.get("feedback") or payload.get("reflective_evidence_summary") or ""),
                    old_validation_score=_optional_float(payload.get("old_validation_score") or payload.get("old_score")),
                    new_validation_score=_optional_float(payload.get("new_validation_score") or payload.get("new_score")),
                )
            )
    return records


def select_distilled_pairs(
    pairs: list[TeacherPair],
    *,
    predictor_name: str | None = None,
    max_pairs: int = 5,
    similarity_threshold: float = 0.86,
) -> list[TeacherPair]:
    scoped = [pair for pair in pairs if predictor_name is None or pair.predictor_name == predictor_name]
    if not scoped and predictor_name is not None:
        scoped = pairs

    ranked = sorted(
        [pair for pair in scoped if pair.validation_delta is not None and abs(pair.validation_delta) > 1e-9],
        key=lambda pair: abs(pair.validation_delta or 0.0),
        reverse=True,
    )
    selected: list[TeacherPair] = []
    for pair in ranked:
        if any(_pair_similarity(pair, kept) >= similarity_threshold for kept in selected):
            continue
        selected.append(pair)
        if len(selected) >= max_pairs:
            break
    return selected


def build_learned_guide(pairs: list[TeacherPair]) -> str:
    lines = [
        "Learned validation-teacher guide for GEPA prompt selection.",
        "Use these distilled old-vs-new cases as weak guidance, not as current validation evidence.",
        "Prefer prompt edits that fix concrete minibatch failures while preserving task constraints.",
        "",
        "Distilled cases:",
    ]
    for idx, pair in enumerate(pairs, start=1):
        lines.extend(
            [
                f"Case {idx}: teacher prefers {pair.preferred_prompt}; validation_delta={pair.validation_delta}",
                f"Predictor: {pair.predictor_name}",
                "Old prompt:",
                _truncate(pair.old_prompt, 520),
                "New prompt:",
                _truncate(pair.new_prompt, 520),
                "Feedback summary:",
                _truncate(pair.feedback, 520),
                "",
            ]
        )
    return "\n".join(lines).strip()


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pair_similarity(a: TeacherPair, b: TeacherPair) -> float:
    a_text = f"{a.old_prompt}\n{a.new_prompt}"
    b_text = f"{b.old_prompt}\n{b.new_prompt}"
    return max(
        SequenceMatcher(None, a_text[:1200], b_text[:1200]).ratio(),
        _jaccard(_tokens(a_text), _tokens(b_text)),
    )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

