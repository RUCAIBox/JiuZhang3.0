import os
import json
import datasets
import pandas as pd


def load_jsonl(load_path, max_samples=None):
    samples = []
    with open(load_path, "r", encoding="utf-8") as f:
        for s in map(json.loads, f):
            if max_samples is not None and len(samples) == max_samples:
                return samples
            samples.append(s)
    return samples


def load_jsonl_ml(load_path):
    samples = []
    with open(load_path, "r", encoding="utf-8") as f:
        sample_str = ""
        for line in f:
            sample_str += line
            if line.startswith("}"):
                samples.append(json.loads(sample_str))
                sample_str = ""
    return samples


def save_jsonl(save_path, samples, indent=None):
    with open(save_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False, indent=indent) + "\n")


def save_jsonl_dataset(data_name, save_dir, samples):
    save_jsonl(os.path.join(save_dir, f"{data_name}.json"), samples)
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=samples))
    dataset.save_to_disk(os.path.join(save_dir, data_name))

def save_dataset(data_name, save_dir, samples):
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=samples))
    dataset.save_to_disk(os.path.join(save_dir, data_name))