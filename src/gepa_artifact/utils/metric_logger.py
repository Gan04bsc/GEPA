import os
import json
import threading
import os
import queue

from ..benchmarks.dspy_program import LangProBeDSPyMetaProgram

class CounterWithLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.step_counter = 0
        self.train_counter = 0
        self.val_counter = 0
        self.test_counter = 0
        self.counter_lock = threading.Lock()
    
    def increment_and_return_state(self, example_split):
        with self.counter_lock:
            self.step_counter += 1
            if example_split == 'train':
                self.train_counter += 1
            elif example_split == 'val':
                self.val_counter += 1
            elif example_split == 'test':
                self.test_counter += 1
            
            return (self.step_counter, self.train_counter, self.val_counter, self.test_counter)

class MetricWithLogger(LangProBeDSPyMetaProgram):
    def __init__(
        self, metric_fn, run_dir, counter_with_lock: CounterWithLock,
        train_dataset=None, val_dataset=None, test_dataset=None,
        log_trace=False, log_example=False, log_prediction=False
    ):
        self.metric_fn = metric_fn
        self.run_dir = run_dir

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.log_trace = log_trace
        self.log_example = log_example
        self.log_prediction = log_prediction
        
        self.counter_with_lock = counter_with_lock
        
        # Single queue and writer
        self._stop_event = threading.Event()
        self.log_queue = queue.Queue()
        self.writer_thread = threading.Thread(target=self._log_writer, daemon=True)
        self.writer_thread.start()
        self._file_handles = {}   # to hold file handles per split
    
    def _get_log_file(self, split):
        """Ensure file handle for each split."""
        subdir = os.path.join(self.run_dir, f"metric_logs")
        os.makedirs(subdir, exist_ok=True)
        file_path = os.path.join(subdir, f"{split}.jsonl")
        if split not in self._file_handles:
            self._file_handles[split] = open(file_path, "a")
        return self._file_handles[split]

    def _log_writer(self):
        while not self._stop_event.is_set() or not self.log_queue.empty():
            try:
                log_item = self.log_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # Build JSON object
            ex, pr, tr, ou, example_split, idx_in_split, counter_vals = log_item
            me = None
            if example_split == 'train':
                me = counter_vals[1]
            elif example_split == 'val':
                me = counter_vals[2]
            elif example_split == 'test':
                me = counter_vals[3]
            data = {
                "example": ex,
                "prediction": pr,
                "metric_output": ou,
                "metric_call_count": me,
                "trace": tr,
                "step_counter": counter_vals[0],
                "train_counter": counter_vals[1],
                "val_counter": counter_vals[2],
                "test_counter": counter_vals[3],
                "idx_in_split": idx_in_split
            }
            if not self.log_trace:
                data.pop("trace", None)
            if not self.log_example:
                data.pop("example", None)
            if not self.log_prediction:
                data.pop("prediction", None)
            # Write to the jsonl file for the split
            if example_split is not None:
                f = self._get_log_file(example_split)
                def fff(x):
                    try:
                        return {**x}
                    except:
                        return repr(x)
                f.write(json.dumps(data, default=fff) + "\n")
                f.flush()
            self.log_queue.task_done()
        # On exit, flush and close all files
        for f in self._file_handles.values():
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Signal log thread to finish
        self._stop_event.set()
        self.writer_thread.join()

    def log_fire_and_forget(self, example, pred, trace, output, example_split, idx_in_split, counter_vals):
        # Just put the data in the queue
        self.log_queue.put((example, pred, trace, output, example_split, idx_in_split, counter_vals))

    def forward(self, example, pred, trace=None):
        example_split = None
        idx_in_split = None
        if self.train_dataset and example in self.train_dataset:
            example_split = 'train'
            idx_in_split = self.train_dataset.index(example)
        elif self.val_dataset and example in self.val_dataset:
            example_split = 'val'
            idx_in_split = self.val_dataset.index(example)
        elif self.test_dataset and example in self.test_dataset:
            example_split = 'test'
            idx_in_split = self.test_dataset.index(example)

        output = self.metric_fn(example, pred, trace)
        counter_vals = self.counter_with_lock.increment_and_return_state(example_split)
        self.log_fire_and_forget(example, pred, trace, output, example_split, idx_in_split, counter_vals)
        return output

    def __call__(self, example, pred, trace=None):
        return self.forward(example, pred, trace)
