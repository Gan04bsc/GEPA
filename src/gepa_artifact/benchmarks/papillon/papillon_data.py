import os

import dspy
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from dspy.datasets import DataLoader
from ..benchmark import Benchmark


def _resolve_hf_dataset_url(filename: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/Columbia-NLP/PUPA/resolve/main/{filename}"


def _get_local_arrow_dataset(config_name: str) -> DatasetDict | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "Columbia-NLP___pupa" / config_name / "0.0.0"
    if not cache_root.exists():
        return None

    candidates = sorted(cache_root.glob("*/pupa-train.arrow"))
    if not candidates:
        return None

    return DatasetDict(train=Dataset.from_file(str(candidates[-1])))

class Papillon(Benchmark):
    def init_dataset(self):
        pupa_tnb = _get_local_arrow_dataset("pupa_tnb")
        if pupa_tnb is None:
            pupa_tnb = load_dataset("csv", data_files=_resolve_hf_dataset_url("PUPA_TNB.csv"))

        pupa_new = _get_local_arrow_dataset("pupa_new")
        if pupa_new is None:
            pupa_new = load_dataset("csv", data_files=_resolve_hf_dataset_url("PUPA_New.csv"))

        examples = [
            dspy.Example(
                {"target_response": x["target_response"], "user_query": x["user_query"], "pii_str": x["pii_units"]}
            ).with_inputs("user_query")
            for x in pupa_new["train"]
        ]

        num_train = 111
        num_val = 111
        num_test = 221

        trainset, testset = examples[:num_train + num_val], examples[num_train + num_val:num_train + num_val + num_test]
        assert len(trainset) == num_train + num_val, f"Expected 500 training examples, but got {len(trainset)}. Total len: {len(examples)}"
        assert len(testset) == num_test, f"Expected 500 validation examples, but got {len(testset)}. Total len: {len(examples)}"
        
        self.dataset = trainset + testset

        self.train_set = trainset[:num_train]
        self.val_set = trainset[num_train:]
        self.test_set = testset

        assert len(self.dataset) == len(trainset) + len(testset), f"Dataset length mismatch: {len(self.dataset)} != {len(trainset) + len(testset)}"
        assert len(self.train_set) == num_train, f"Train set length mismatch: {len(self.train_set)} != {num_train}"
        assert len(self.val_set) == num_val, f"Validation set length mismatch: {len(self.val_set)} != {num_val}"
        assert len(self.test_set) == num_test, f"Test set length mismatch: {len(self.test_set)} != {num_test}"
