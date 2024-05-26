import argparse
import datasets
import random

random.seed(42)
import os
from datasets import load_dataset, load_from_disk
from src.utils.file_utils import load_jsonl


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


import re


def split_latex_sections(latex_content):
    pattern = r"\\section{.*?}"
    sections = re.split(pattern, latex_content)
    return [section.strip() for section in sections if section.strip()]


def build_prompt(x, prompt, style="college"):
    """Build the prompt based on the generation type"""
    save_prompts = []
    save_types = []
    for i, text in enumerate(x["text"]):
        # Split by LaTeX `\section{xx}`
        sections = separate_sections(text)
        for section in sections:
            snippet = section.strip()
            snippet = snippet[: min(len(snippet), EXTRACT_SIZE)]
            chosen_key = random.choice(list(prompt.keys()))
            cur_prompt = prompt[chosen_key]
            cur_prompt = cur_prompt.replace("<EXTRACT>", f"```\n{snippet}\n```")
            save_prompts.append(cur_prompt)
            save_types.append(chosen_key)

    return {f"prompt_{style}": save_prompts, "prompt_type": save_types}


def format_sample(sample):
    return {"text": sample["text"], "subset": sample["subset"]}


def separate_sections(markdown_text):
    # Regular expression pattern to match section headers
    section_pattern = re.compile(r"^(##)\s+(.*?)$", re.MULTILINE)

    # Find all section headers in the markdown text
    sections = []
    prev_index = 0
    for match in section_pattern.finditer(markdown_text):
        # Extract the section level and title
        level = len(match.group(1))
        title = match.group(2)

        # Extract the section content
        start_index = match.start()
        content = markdown_text[prev_index:start_index].strip()

        # Append the previous section to the list
        if prev_index != 0:
            sections.append(content)

        # Update the previous index
        prev_index = start_index

        # Print the section information
        # print(f"Section Level: {level}")
        # print(f"Section Title: {title}")
        # print("---")

    # Append the last section
    last_section = markdown_text[prev_index:].strip()
    if last_section:
        sections.append(last_section)

    return sections


if __name__ == "__main__":
    args = get_args()

    print(f"Loading MathPile textbooks data...")
    all_data = []
    data_path = os.path.join(args.data_dir, "textbooks", "textbooks_markdown.jsonl")
    ds = load_dataset(
        "json", data_files=data_path, split="train", cache_dir=args.cache_dir
    )
    all_data.append(ds)
    data_path = os.path.join(
        args.data_dir, "textbooks", "synthetic_textbooks_markdown.jsonl"
    )
    ds = load_dataset(
        "json", data_files=data_path, split="train", cache_dir=args.cache_dir
    )
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
