from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Iterable

import dspy
from datasets import Dataset, concatenate_datasets, load_dataset

from ..benchmark import Benchmark

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOTPOT_QA_DIR = REPO_ROOT / "data" / "hotpotqa" / "qa"
DEFAULT_HOTPOT_VARIANT = "fullwiki"
DEFAULT_TRAIN_VAL_SPLIT_SEED = 13
DEFAULT_HELDOUT_VAL_SIZE = 5000
DEFAULT_HF_DATASETS_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "datasets"


class HotpotQABench(Benchmark):
    """HotpotQA benchmark with an evaluable paper-aligned split protocol.

    Adopted protocol:
    - train: official HotpotQA train minus a fixed held-out validation slice
    - val: deterministic held-out slice from official HotpotQA train
    - test: official HotpotQA validation split

    Rationale:
    - `GEPA.pdf` reports HotpotQA with 150 train / 300 validation / 300 test.
    - The public HF `hotpot_qa/fullwiki` test split has no labels, so it is not
      directly usable for artifact metric-based evaluation.
    - The repository-wide Benchmark base class still applies its trim policy
      after these explicit splits are created; that trim yields the paper counts.
    """

    def init_dataset(self):
        train_rows, test_rows, source = self._load_rows()
        train_examples = self._to_examples(train_rows)
        test_examples = self._to_examples(test_rows)
        self.train_set, self.val_set = self._split_train_and_val(train_examples)
        self.test_set = test_examples
        self.dataset = [*self.train_set, *self.val_set, *self.test_set]
        self.dataset_source = source
        self.split_protocol = {
            "train_source": "official_train_minus_fixed_heldout_validation",
            "val_source": "fixed_heldout_slice_from_official_train",
            "test_source": "official_validation_split",
            "split_seed": self._split_seed(),
            "heldout_val_size": self._heldout_val_size(len(train_examples)),
            "dataset_variant": self._dataset_variant(),
            "data_dir": str(self._qa_dir()),
        }

    @staticmethod
    def _qa_dir() -> Path:
        override = os.getenv("HOTPOTQA_QA_DIR")
        return Path(override).expanduser().resolve() if override else DEFAULT_HOTPOT_QA_DIR

    @staticmethod
    def _dataset_variant() -> str:
        return os.getenv("HOTPOTQA_DATASET_VARIANT", DEFAULT_HOTPOT_VARIANT)

    @staticmethod
    def _split_seed() -> int:
        return int(os.getenv("HOTPOTQA_SPLIT_SEED", str(DEFAULT_TRAIN_VAL_SPLIT_SEED)))

    @staticmethod
    def _heldout_val_size(train_size: int) -> int:
        requested = int(os.getenv("HOTPOTQA_HELDOUT_VAL_SIZE", str(DEFAULT_HELDOUT_VAL_SIZE)))
        return max(1, min(requested, max(1, train_size - 1)))

    def _load_rows(self) -> tuple[list[dict], list[dict], str]:
        local_train = self._first_existing(
            self._qa_dir(),
            [
                "hotpot_train_v1.1.json",
                "train.json",
                "train.jsonl",
            ],
        )
        local_validation = self._first_existing(
            self._qa_dir(),
            [
                "hotpot_dev_fullwiki_v1.json",
                "validation.json",
                "dev.json",
                "dev_fullwiki.json",
                "validation.jsonl",
            ],
        )

        if local_train and local_validation:
            return (
                self._load_local_records(local_train),
                self._load_local_records(local_validation),
                f"local:{local_train.name}+{local_validation.name}",
            )

        if local_train or local_validation:
            warnings.warn(
                "HotpotQA local data directory is partially populated. Falling back to "
                "Hugging Face hotpot_qa/fullwiki for a consistent source of train/test rows.",
                stacklevel=2,
            )

        cached_rows = self._load_cached_hf_rows()
        if cached_rows is not None:
            return cached_rows

        raw_datasets = load_dataset("hotpot_qa", self._dataset_variant(), trust_remote_code=True)
        return list(raw_datasets["train"]), list(raw_datasets["validation"]), "hf:hotpot_qa/fullwiki"

    @classmethod
    def _load_cached_hf_rows(cls) -> tuple[list[dict], list[dict], str] | None:
        dataset_variant = cls._dataset_variant()
        for cache_root in cls._datasets_cache_dir_candidates():
            variant_root = cache_root / "hotpot_qa" / dataset_variant / "0.0.0"
            if not variant_root.exists():
                continue

            for fingerprint_dir in sorted(variant_root.iterdir(), reverse=True):
                if not fingerprint_dir.is_dir():
                    continue
                train_shards = sorted(fingerprint_dir.glob("hotpot_qa-train-*.arrow"))
                validation_arrow = fingerprint_dir / "hotpot_qa-validation.arrow"
                if not train_shards or not validation_arrow.exists():
                    continue

                train_dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in train_shards])
                validation_dataset = Dataset.from_file(str(validation_arrow))
                return (
                    list(train_dataset),
                    list(validation_dataset),
                    f"hf_cache_arrow:{fingerprint_dir}",
                )

        return None

    @staticmethod
    def _datasets_cache_dir_candidates() -> list[Path]:
        candidates: list[Path] = []

        hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
        if hf_datasets_cache:
            candidates.append(Path(hf_datasets_cache).expanduser().resolve())

        hf_home = os.getenv("HF_HOME")
        if hf_home:
            candidates.append((Path(hf_home).expanduser() / "datasets").resolve())

        candidates.append(DEFAULT_HF_DATASETS_CACHE_DIR)

        deduped: list[Path] = []
        seen = set()
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str in seen:
                continue
            seen.add(candidate_str)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _first_existing(directory: Path, filenames: Iterable[str]) -> Path | None:
        for filename in filenames:
            candidate = directory / filename
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_local_records(path: Path) -> list[dict]:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]

        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Expected a JSON array in {path}, got {type(payload).__name__}.")

    @staticmethod
    def _to_examples(rows: Iterable[dict]) -> list[dspy.Example]:
        return [dspy.Example(**row).with_inputs("question") for row in rows]

    def _split_train_and_val(self, train_examples: list[dspy.Example]) -> tuple[list[dspy.Example], list[dspy.Example]]:
        if len(train_examples) < 2:
            raise ValueError("HotpotQA train split must contain at least two examples to create train/val splits.")

        rng = random.Random(self._split_seed())
        indices = list(range(len(train_examples)))
        rng.shuffle(indices)

        heldout_val_size = self._heldout_val_size(len(train_examples))
        val_indices = set(indices[:heldout_val_size])
        val_set = [train_examples[idx] for idx in indices if idx in val_indices]
        train_set = [train_examples[idx] for idx in indices if idx not in val_indices]
        return train_set, val_set
