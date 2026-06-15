from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class JudgeDecision:
    preferred_prompt: str
    confidence: float
    short_reason: str
    risk_note: str = "none"
    old_score: float | None = None
    new_score: float | None = None

    @property
    def score_delta(self) -> float | None:
        if self.old_score is None or self.new_score is None:
            return None
        return self.new_score - self.old_score


class JudgeClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return raw model text for a judge prompt."""


def build_selection_prompt(
    *,
    predictor_name: str,
    old_prompt: str,
    new_prompt: str,
    feedback: str,
    version: str,
    learned_guide: str | None = None,
) -> str:
    score_keys = ""
    if version in {"v1", "combined"}:
        score_keys = '  "old_score": float between 0 and 100,\n  "new_score": float between 0 and 100,\n'
    guide_block = f"\nLearned guide:\n{learned_guide}\n" if learned_guide else ""
    if version.startswith("v5") and learned_guide:
        guide_block = f"\nWarmup rules library and optional calibration examples:\n{learned_guide}\n"
    return f"""You are a GEPA prompt-selection judge.

Compare OLD and NEW instructions for the target predictor. Use only the prompt diff, minibatch feedback, and optional learned guide.
For v5 rules protocols, treat the rules library as weak policy distilled from warmup validation. It is not current validation evidence.
Return exactly one JSON object:
{{
{score_keys}  "preferred_prompt": "old" or "new",
  "confidence": float between 0 and 1,
  "short_reason": "short explanation",
  "risk_note": "short caveat"
}}

Target predictor: {predictor_name}

OLD instruction:
{old_prompt}

NEW instruction:
{new_prompt}

Minibatch feedback:
{feedback}
{guide_block}
"""


def parse_judge_decision(raw_text: str) -> JudgeDecision:
    payload = _extract_json(raw_text)
    preferred = str(payload.get("preferred_prompt", "")).lower().strip()
    if preferred not in {"old", "new"}:
        raise ValueError("judge preferred_prompt must be 'old' or 'new'.")
    confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
    return JudgeDecision(
        preferred_prompt=preferred,
        confidence=confidence,
        short_reason=str(payload.get("short_reason", "")).strip(),
        risk_note=str(payload.get("risk_note", "none")).strip() or "none",
        old_score=_optional_float(payload.get("old_score")),
        new_score=_optional_float(payload.get("new_score")),
    )


def combined_delta(
    *,
    old_validation_score: float,
    new_validation_score: float,
    judge_decision: JudgeDecision,
    validation_weight: float = 1.0,
    judge_weight: float = 1.0,
    normalize_validation_delta: bool = True,
) -> float:
    validation_delta = new_validation_score - old_validation_score
    if normalize_validation_delta:
        validation_delta /= 100.0
    signed_confidence = judge_decision.confidence if judge_decision.preferred_prompt == "new" else -judge_decision.confidence
    return validation_weight * validation_delta + judge_weight * signed_confidence


def _extract_json(raw_text: str) -> dict:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in judge response.")
    return json.loads(raw_text[start : end + 1])


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
