import argparse
import datasets
from datasets import load_dataset
from src.utils.file_utils import load_jsonl

datasets.disable_caching()
import random


def shuffle(data: datasets.Dataset):
    print("Convert to list")
    data = data.to_list()
    print("Shuffle")
    random.shuffle(data)
    print("Convert back")
    return datasets.Dataset.from_list(data)


def merge_two_stage_train_data(stage1_paths, stage2_paths, save_path):
    print("Stage 1")
    all_samples = []
    if len(stage1_paths):
        for i, data_path in enumerate(stage1_paths):
            ds = load_dataset(data_path, split="train")
            all_samples.append(ds)
        stage1_ds = datasets.concatenate_datasets(all_samples)
        del all_samples
        stage1_ds = shuffle(stage1_ds)
    if len(stage2_paths):
        print("Stage 2")
        all_samples = []
        for i, data_path in enumerate(stage2_paths):
            ds = load_dataset(data_path, split="train")
            all_samples.append(ds)
        stage2_ds = datasets.concatenate_datasets(all_samples)
        stage2_ds = shuffle(stage2_ds)
    if len(stage1_paths) and len(stage2_paths):
        ds = datasets.concatenate_datasets([stage1_ds, stage2_ds])
    else:
        ds = stage1_ds if len(stage1_paths) else stage2_ds
    print(save_path)
    ds.save_to_disk(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    merge_two_stage_train_data(
        [
            "ToheartZhang/JiuZhang3.0-Corpus-PT-CoT",
            "ToheartZhang/JiuZhang3.0-Corpus-PT-Tool",
        ],
        ["ToheartZhang/JiuZhang3.0-Corpus-SFT"],
        args.save_path,
    )
