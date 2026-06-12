import os
from pathlib import Path

from ..benchmark import Benchmark
import dspy

from datasets import Dataset, load_dataset


def _load_cached_train_split(repo_id: str):
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    dataset_cache_root = hf_home / "datasets" / repo_id.replace("/", "___")
    if not dataset_cache_root.exists():
        return None

    arrow_candidates = sorted(dataset_cache_root.glob("**/*train.arrow"))
    if not arrow_candidates:
        return None

    return Dataset.from_file(str(arrow_candidates[-1]))


def _load_train_split(repo_id: str):
    cached_split = _load_cached_train_split(repo_id)
    if cached_split is not None:
        return cached_split
    return load_dataset(repo_id)["train"]

class AIMEBench(Benchmark):
    def init_dataset(self):
        train_split = _load_train_split("AI-MO/aimo-validation-aime")
        train_split = [
            dspy.Example({
                "problem": x['problem'],
                'solution': x['solution'],
                'answer': x['answer'],
            }).with_inputs("problem")
            for x in train_split
        ]
        import random
        random.Random(0).shuffle(train_split)
        tot_num = len(train_split)

        test_split = _load_train_split("MathArena/aime_2025")
        test_split = [
            dspy.Example({
                "problem": x['problem'],
                'answer': x['answer'],
            }).with_inputs("problem")
            for x in test_split
        ]

        # Paper protocol: use AIME 2025 as the final test set with 5 repeated
        # evaluations per problem, i.e. 30 unique problems expanded to 150 test
        # instances so repeated stochastic samples are visible at evaluation time.
        repeated_test_split = test_split * 5

        self.train_set = train_split[:int(0.5 * tot_num)]
        self.val_set = train_split[int(0.5 * tot_num):]
        self.test_set = repeated_test_split

        # Keep the aggregate dataset consistent with the explicit train/val/test
        # splits above. The paper protocol only repeats the 30 AIME-2025 test
        # problems 5 times in total, yielding a 150-example final test set.
        self.dataset = self.train_set + self.val_set + self.test_set
        self.dataset_source = {
            "train_val_source": "AI-MO/aimo-validation-aime",
            "test_source": "MathArena/aime_2025",
        }
        self.split_protocol = {
            "train_val_source_years": "AIME 2022-2024",
            "test_source_year": "AIME 2025",
            "train_size": len(self.train_set),
            "val_size": len(self.val_set),
            "test_size": len(self.test_set),
            "test_unique_problem_count": len(test_split),
            "test_repeats_per_problem": 5,
            "paper_alignment_note": "AIME-2025 questions are repeated 5 times for final evaluation.",
        }
        # Repeated test questions must not hit the LM cache during final
        # evaluation; otherwise the 5 repeats collapse into 1 sampled answer.
        self.disable_cache_for_final_evaluation = True
