import argparse
import os
import torch
import datasets
from tqdm import tqdm
from src.utils.file_utils import load_jsonl, save_jsonl
from src.common.templates import INFERENCE_PATTERN


def merge_data(paths, save_path):
    all_samples = []
    for p in paths:
        if "json" in p:
            samples = load_jsonl(p)
        else:
            samples = datasets.load_from_disk(p)
            # samples = samples.to_list()
        print(samples)
        samples = samples.select_columns(["prompt_mix", "prompt_type"])
        samples = samples.add_column(
            "source", [p.split("/")[-1].split("-all_mix")[0]] * len(samples)
        )
        print(samples)
        all_samples.append(samples)
    print("Concanate")
    ds = datasets.concatenate_datasets(all_samples)
    print("Concanate end")
    ds.save_to_disk(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()
    merge_data(
        [
            os.path.join(args.data_path, "stack-mix-0"),
            os.path.join(args.data_path, "wiki-mix-0"),
            os.path.join(args.data_path, "arxiv-mix-0.60-1.00-0"),
            os.path.join(args.data_path, "textbook-mix-0"),
            os.path.join(args.data_path, "web-mix-0.20-1.00-0"),
        ],
        os.path.join(args.data_path, "merged"),
    )
