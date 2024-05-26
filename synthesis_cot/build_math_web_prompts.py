import argparse
import datasets
import random

random.seed(2025)
import os
from datasets import load_dataset, load_from_disk



# EXTRACT_SIZE = 10000
EXTRACT_SIZE = 1000
LEVELS = ["college", "middle", "grade", "high", "amc8", "amc10", "amc12", "aime"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="HuggingFaceTB/auto_math")
    parser.add_argument("--generation_style", type=str, default="college")
    parser.add_argument("--run_all_styles", action="store_true")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--scope", type=str, default="0.50-0.95")
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=16)
    parser.add_argument("--data_source", default="web")
    return parser.parse_args()


def get_extract_size(text, word_cnt=1000):
    words = text.split()
    if len(words) <= word_cnt:
        return len(text)
    else:
        total_chars = sum(len(word) for word in words[:word_cnt])
        spaces = word_cnt - 1
        estimated_length = total_chars + spaces
        return estimated_length


def build_prompt(x, prompt, style="college"):
    """Build the prompt based on the generation type"""
    snippet = x["text"].strip()
    # print(get_extract_size(snippet))
    # if get_extract_size(snippet) < 1000:
    #     print(snippet)
    #     return {"prompt": None, "prompt_type": None}
    # snippet = snippet[: min(len(snippet), min(EXTRACT_SIZE, get_extract_size(snippet)))] TODO is this a bug?
    snippet = snippet[: min(len(snippet), EXTRACT_SIZE)]
    # print(len(snippet))
    if isinstance(prompt, dict):
        chosen_key = random.choice(list(prompt.keys()))
        prompt = prompt[chosen_key]
    prompt = prompt.replace("<EXTRACT>", snippet).strip()
    # if "## New Problem" not in prompt:
    #     prompt += "\n\n## New Problem"
    return {f"prompt_mix": prompt, "prompt_type": chosen_key}


if __name__ == "__main__":
    args = get_args()

    print(f"Loading AutoMathText web data...")
    scope_start, scope_end = args.scope.split("-")
    ds = load_dataset("math-ai/AutoMathText", f"web-{scope_start}-to-{scope_end}")
    print("Sample size", ds)
    if args.generation_style != "mix":
        suffix = f"_{args.generation_style}"
        print(f"📖 Building prompts with a {args.generation_style}...")
        with open(args.prompt_path, "r") as f:
            prompt = f.read()
        ds = ds.map(
            build_prompt,
            num_proc=48,
            fn_kwargs={"prompt": prompt, "style": args.generation_style},
            load_from_cache_file=False,
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
            num_proc=48,
            fn_kwargs={"prompt": prompts, "style": args.generation_style},
            load_from_cache_file=False,
        )
        print(ds)
    print(ds)
    ds = ds.filter(lambda x: x["prompt_mix"] is not None, num_proc=96)
    print(ds[0][f"prompt_{args.generation_style}"])

    # print(ds[0][f"prompt"])
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
        save_path = f"{args.save_path}{suffix}-{args.scope}-{i}"
        ds_chunk.save_to_disk(save_path)
        print(f"✅ Data available at {save_path}!")
    print("Done!")
