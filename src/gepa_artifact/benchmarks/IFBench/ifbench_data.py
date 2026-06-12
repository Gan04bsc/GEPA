import os
from pathlib import Path
from ..benchmark import Benchmark
import dspy
import json
from datasets import load_dataset


DEFAULT_IFBENCH_DATA_DIR = Path.home() / ".cache" / "gepa" / "ifbench"
IFBENCH_REPO_ID = "microsoft/IFBench"


def _data_dir() -> Path:
    return Path(os.environ.get("GEPA_IFBENCH_DATA_DIR", DEFAULT_IFBENCH_DATA_DIR)).expanduser().resolve()


def _load_jsonl(path: Path):
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            examples.append(dspy.Example(**json.loads(line)).with_inputs("prompt"))
    return examples


def _load_split(split_name: str, filename: str):
    path = _data_dir() / filename
    if path.exists():
        return _load_jsonl(path)
    try:
        dataset = load_dataset(IFBENCH_REPO_ID, split=split_name)
        return [dspy.Example(**dict(row)).with_inputs("prompt") for row in dataset]
    except Exception as exc:
        raise FileNotFoundError(
            f"IFBench data file not found: {path}. Set GEPA_IFBENCH_DATA_DIR to a directory "
            f"containing {filename}, or make sure Hugging Face dataset {IFBENCH_REPO_ID} is available."
        ) from exc

class IFBench(Benchmark):
    def init_dataset(self):
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        self.test_set = _load_split("test", "IFBench_test.jsonl")
        train_val_set = _load_split("train", "IFBench_train.jsonl")

        self.train_set = train_val_set[300:600]
        self.val_set = train_val_set[:300]

        self.dataset = self.train_set + self.val_set + self.test_set
