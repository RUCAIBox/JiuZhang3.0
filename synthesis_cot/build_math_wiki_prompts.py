import argparse
import datasets
import random

random.seed(42)
import os
from datasets import load_dataset, load_from_disk


EXTRACT_SIZE = 1000
LEVELS = ["college", "middle", "grade", "high", "amc8", "amc10", "amc12", "aime"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="HuggingFaceTB/auto_math")
    parser.add_argument("--generation_style", type=str, default="college")
    parser.add_argument("--run_all_styles", action="store_true")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--scope", type=str, default="0.50-0.95")
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=16)
    parser.add_argument("--data_source", default="web")
    return parser.parse_args()


def build_prompt(x, prompt, style="college"):
    """Build the prompt based on the generation type"""
    save_prompts = []
    save_types = []
    save_texts = []
    save_subsets = []
    for i, text in enumerate(x["text"]):
        snippet = text.strip()
        snippet = snippet[: min(len(snippet), EXTRACT_SIZE)]
        # if isinstance(prompt, dict):
        #     chosen_key = random.choice(list(prompt.keys()))
        #     prompt = prompt[chosen_key]
        for key in prompt:
            cur_prompt = prompt[key]
            cur_prompt = cur_prompt.replace("<EXTRACT>", snippet)
            save_prompts.append(cur_prompt)
            save_types.append(key)
            save_texts.append(text)
            save_subsets.append(x["subset"][i])
    return {f"prompt_{style}": save_prompts, "prompt_type": save_types}


def format_sample(sample):
    return {"text": sample["text"], "subset": sample["subset"]}


if __name__ == "__main__":
    args = get_args()

    print(f"Loading AutoMathText wiki data...")
    all_data = []
    data_path = os.path.join(
        args.data_dir, "wikipedia", "wikipedia_en_mathematics_nopic_2023-08_v0.2.jsonl"
    )
    ds = load_dataset(
        "json", data_files=data_path, split="train", cache_dir=args.cache_dir
    )
    ds = ds.map(format_sample, remove_columns=ds.column_names)
    all_data.append(ds)
    data_path = os.path.join(
        args.data_dir, "proofwiki", "ProofWiki_theorem_proofs.jsonl"
    )
    ds = load_dataset(
        "json", data_files=data_path, split="train", cache_dir=args.cache_dir
    )
    ds = ds.map(format_sample, remove_columns=ds.column_names)
    all_data.append(ds)
    data_path = os.path.join(args.data_dir, "proofwiki", "ProofWiki_definitions.jsonl")
    ds = load_dataset(
        "json", data_files=data_path, split="train", cache_dir=args.cache_dir
    )
    ds = ds.map(format_sample, remove_columns=ds.column_names)
    all_data.append(ds)
    ds = datasets.concatenate_datasets(all_data)
    print("Sample size", ds)
    if args.generation_style != "mix":
        suffix = f"_{args.generation_style}"
        print(f"📖 Building prompts with a {args.generation_style}...")
        with open(args.prompt_path, "r") as f:
            prompt = f.read()
        ds = ds.map(
            build_prompt,
            batched=True,
            num_proc=48,
            fn_kwargs={"prompt": prompt, "style": args.generation_style},
            load_from_cache_file=False,
            remove_columns=ds.column_names,
        )
        print(ds)
    else:
        suffix = f"_{args.generation_style}"
        print(f"📖 Building prompts with a {args.generation_style}...")
        prompts = {}
        for level in LEVELS:
            with open(os.path.join(args.prompt_path, f"{level}.md"), "r") as f:
                prompts[level] = f.read().strip()
        ds = ds.map(
            build_prompt,
            batched=True,
            num_proc=48,
            fn_kwargs={"prompt": prompts, "style": args.generation_style},
            load_from_cache_file=False,
            remove_columns=ds.column_names,
        )
        print(ds)
    print(ds)
    print(ds[0][f"prompt_{args.generation_style}"])
    print("-" * 100)
    # save_path = f"{args.save_path}{suffix}-{args.scope}"
    # ds.save_to_disk(save_path)
    # print(f"✅ Data available at {save_path}!")
    # split the data into chunks
    num_chunks = args.num_chunks
    chunk_size = len(ds) // num_chunks
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else len(ds)
        ds_chunk = ds.select(range(start, end))
        save_path = f"{args.save_path}{suffix}-{i}"
        ds_chunk.save_to_disk(save_path)
        print(f"✅ Data available at {save_path}!")
    print("Done!")
