from datasets import load_dataset
from datasets import Dataset
import random
import re
import json
import os
from tqdm import tqdm

# Take proofwiki as an example
file_paths = []
for root, dirs, files in os.walk("/path/to/AutoMathText/proofwiki"):
    for file in files:
        if file.endswith(".jsonl"):
            file_paths.append(os.path.join(root, file))

file_paths = []
for root, dirs, files in os.walk("/path/to/AutoMathText/proofwiki"):
    for file in files:
        if file.endswith(".jsonl"):
            file_paths.append(os.path.join(root, file))
proofwiki = []
for file_path in tqdm(file_paths):
    with open(file_path, "r") as f:
        data = f.readlines()
        for line in data:
            sample = json.loads(line)
            text = sample["text"][:2000]
            if text[-1] != " " and text[-1] != "\n":
                text = text[:max(text.rfind("."), text.rfind("\n"), text.rfind("}"))+1]
            proofwiki.append(text)

seeds = proofwiki

with open("prompts/tool/prompt.md", "r") as f:
    prompt = f.read().strip()

mmos = load_dataset("cyzhh/MMOS")
mmos = mmos.filter(lambda example: 'box' in example["completion"])

total_examples = len(mmos["train"])

dataset = []
for seed in tqdm(seeds):

    random_example = random.randint(0, total_examples - 1)
    sample = mmos["train"][random_example]

    s = prompt.format(seed=seed, question=sample["prompt"].strip("<|user|>\n").strip("\n<|assistant|>"), solution=re.sub(r'    "{3}.*?"{3}\n', '', sample["completion"], flags=re.DOTALL))
    dataset.append({"text": s})

dataset = Dataset.from_list(dataset)
dataset.save_to_disk("/path/to/save")