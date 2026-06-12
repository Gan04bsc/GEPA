from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ValidationSubset:
    retained_fraction: float
    sampling_mode: str
    selected_indices: list[int]


def select_validation_subset(
    examples: Sequence[T],
    *,
    retained_fraction: float,
    sampling_mode: str,
    seed: int,
    iteration: int = 0,
) -> ValidationSubset:
    if not 0.0 <= retained_fraction <= 1.0:
        raise ValueError("retained_fraction must be between 0 and 1.")
    keep = round(len(examples) * retained_fraction)
    indices = list(range(len(examples)))
    if sampling_mode != "fixed":
        raise ValueError("sampling_mode must be 'fixed'.")
    Random(seed).shuffle(indices)
    return ValidationSubset(
        retained_fraction=retained_fraction,
        sampling_mode=sampling_mode,
        selected_indices=indices[:keep],
    )


def validation_accepts_update(old_score: float, new_score: float, *, min_delta: float = 0.0) -> bool:
    return (new_score - old_score) > min_delta
