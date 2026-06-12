import json
import os
import random
from pathlib import Path

import dspy
import tqdm
from datasets import load_dataset

from ..benchmark import Benchmark
from .hover_utils import count_unique_docs

HOVER_TRAIN_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json"
HOVER_TRAIN_PATH_ENV_VARS = ("GEPA_HOVER_TRAIN_PATH", "HOVER_TRAIN_PATH")
DEFAULT_LOCAL_HOVER_TRAIN_PATH = Path.home() / ".cache" / "gepa" / "hover" / "hover_train_release_v1.1.json"


def resolve_hover_train_source():
    for env_var in HOVER_TRAIN_PATH_ENV_VARS:
        value = os.environ.get(env_var)
        if value and value.strip():
            return value.strip()
    if DEFAULT_LOCAL_HOVER_TRAIN_PATH.exists():
        return str(DEFAULT_LOCAL_HOVER_TRAIN_PATH)
    return HOVER_TRAIN_URL


def load_hover_train_examples():
    source = resolve_hover_train_source()
    source_path = Path(source)
    if source_path.exists():
        with source_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    dataset = load_dataset("json", data_files={"train": source})
    return dataset["train"]


class hoverBench(Benchmark):
    def init_dataset(self):
        hf_trainset = load_hover_train_examples()

        reformatted_hf_trainset = []

        for example in tqdm.tqdm(hf_trainset):
            claim = example["claim"]
            supporting_facts = [
                {"key": fact[0], "value": fact[1]}
                for fact in example["supporting_facts"]
            ]
            label = example["label"]

            if len({fact["key"] for fact in supporting_facts}) == 3:  # Limit to 3 hop examples
                reformatted_hf_trainset.append(
                    dict(claim=claim, supporting_facts=supporting_facts, label=label)
                )

        rng = random.Random()
        rng.seed(0)
        rng.shuffle(reformatted_hf_trainset)
        rng = random.Random()
        rng.seed(1)

        trainset = reformatted_hf_trainset

        trainset = [dspy.Example(**x).with_inputs("claim") for x in trainset]

        self.dataset = trainset
