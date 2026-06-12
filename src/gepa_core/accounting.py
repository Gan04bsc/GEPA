from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


BucketName = Literal["optimization", "judge", "minibatch", "validation", "evaluation"]


@dataclass
class UsageBucket:
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, *, input_tokens: int = 0, output_tokens: int = 0, api_calls: int = 0, seconds: float = 0.0) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.api_calls += api_calls
        self.seconds += seconds


@dataclass
class CostAccounting:
    buckets: dict[BucketName, UsageBucket] = field(
        default_factory=lambda: {
            "optimization": UsageBucket(),
            "judge": UsageBucket(),
            "minibatch": UsageBucket(),
            "validation": UsageBucket(),
            "evaluation": UsageBucket(),
        }
    )

    def add(self, bucket: BucketName, **usage: int | float) -> None:
        self.buckets[bucket].add(**usage)

    @property
    def total_tokens(self) -> int:
        return sum(bucket.total_tokens for bucket in self.buckets.values())

    @property
    def total_api_calls(self) -> int:
        return sum(bucket.api_calls for bucket in self.buckets.values())

    @property
    def total_seconds(self) -> float:
        return sum(bucket.seconds for bucket in self.buckets.values())

    def as_dict(self) -> dict[str, int | float | dict[str, int | float]]:
        return {
            "total_tokens": self.total_tokens,
            "total_api_calls": self.total_api_calls,
            "total_seconds": self.total_seconds,
            "buckets": {
                name: {
                    "input_tokens": bucket.input_tokens,
                    "output_tokens": bucket.output_tokens,
                    "total_tokens": bucket.total_tokens,
                    "api_calls": bucket.api_calls,
                    "seconds": bucket.seconds,
                }
                for name, bucket in self.buckets.items()
            },
        }

