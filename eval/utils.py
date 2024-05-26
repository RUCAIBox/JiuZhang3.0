import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any

from examples import get_examples


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


# def load_prompt(data_name, prompt_type):
#     if data_name in ["gsm-hard", "svamp", "tabmwp", "asdiv", "mawps"]:
#         data_name = "gsm8k"
#     if data_name in ["math-oai", "hungarian_exam"]:
#         data_name = "math"
#     if prompt_type in ["platypus_fs"]:
#         prompt_type = "cot"
#     if prompt_type in ["tool-integrated"]:
#         prompt_type = "tora"

#     if prompt_type in ["cot", "pal", "tora", "jiuzhang_fs"]:
#         prompt_path = "./prompts/{}/{}.md".format(prompt_type, data_name)
#         if not os.path.exists(prompt_path):
#             prompt_path = "./prompts/{}.md".format(prompt_type)
#         if os.path.exists(prompt_path):
#             with open(prompt_path, "r", encoding="utf-8") as fp:
#                 prompt = fp.read().strip() + "\n\n"
#         else:
#             print(f"Error: prompt file {prompt_path} not found")
#             prompt = ""
#     else:
#         prompt = ""
#     return prompt


EXAMPLES = get_examples()


def load_prompt(data_name, prompt_type, num_shots):
    if not num_shots:
        return []

    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"

    return EXAMPLES[data_name][:num_shots]


PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integreted": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    "abel": (
        "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
}


def construct_prompt(example, data_name, args):
    demos = load_prompt(data_name, args.prompt_type, args.num_shots)
    prompt_type = args.prompt_type
    if prompt_type == "platypus_fs":
        prompt_type = "cot"
    if prompt_type == "tool-integrated":
        prompt_type = "tora"

    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    demo_prompt = splitter.join(
        [
            input_template.format(input=q) + output_template.format(output=a)
            for q, a in demos
        ]
    )
    context = input_template.format(input=example["question"])
    if len(demo_prompt):
        full_prompt = demo_prompt + splitter + context
    else:
        full_prompt = context

    if args.prompt_type == "platypus_fs":
        full_prompt_temp = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt_temp.format(instruction=full_prompt)

    if prompt_type == "tora":
        full_prompt = (
            """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

"""
            + full_prompt
        )

    return full_prompt.strip(" ")   # important!


# def construct_prompt(example, data_name, args):
#     demo_prompt = load_prompt(data_name, args.prompt_type)
#     # Base models
#     if args.prompt_type in ["direct", "cot"]:
#         context = f"Question: {example['question']}\nAnswer:"
#         full_prompt = demo_prompt + context
#     elif args.prompt_type == "pal":
#         context = f"Question: {example['question']}"
#         full_prompt = demo_prompt + context
#     elif args.prompt_type in ["tool-integreted"]:
#         context = f"Question: {example['question']}\n\nSolution:"
#         full_prompt = demo_prompt + context

#     # SFT models
#     elif args.prompt_type in ["self-instruct", "tora"]:
#         full_prompt = f"<|user|>\n{example['question']}\n<|assistant|>\n"
#     elif args.prompt_type in ["self-instruct-boxed"]:
#         full_prompt = f"<|user|>\n{example['question']}\nEnclose the final answer using \\boxed{{}}.\n<|assistant|>\n"
#     elif args.prompt_type == "wizard_zs":
#         full_prompt = (
#             "Below is an instruction that describes a task. "
#             "Write a response that appropriately completes the request.\n\n"
#             "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
#         )
#         full_prompt = full_prompt.format(instruction=example["question"])
#     elif args.prompt_type == "platypus_fs":
#         full_prompt = (
#             "Below is an instruction that describes a task. "
#             "Write a response that appropriately completes the request.\n\n"
#             "### Instruction:\n{instruction}\n\n### Response:\n"
#         )
#         full_prompt = full_prompt.format(
#             instruction=demo_prompt + f"Question: {example['question']}\nAnswer:"
#         )
#     elif args.prompt_type == "deepseek-math":
#         full_prompt = (
#             "User: {instruction}\nPlease reason step by step, "
#             "and put your final answer within \\boxed{{}}.\n\nAssistant:"
#         )
#         full_prompt = full_prompt.format(instruction=example["question"])
#     elif args.prompt_type == "kpmath":
#         full_prompt = (
#             "User: Please reason step by step and put your final answer at the end "
#             'with "The answer is: ".\n\n{instruction}\n\nAssistant:'
#         )
#         full_prompt = full_prompt.format(instruction=example["question"])
#     elif args.prompt_type == "jiuzhang":
#         full_prompt = "## Question\n{instruction}\n\n## Solution\n"
#         full_prompt = full_prompt.format(instruction=example["question"])
#     elif args.prompt_type == "jiuzhang_tora":
#         full_prompt = "## Question\n{instruction}\n\n## Code Solution\n"
#         full_prompt = full_prompt.format(instruction=example["question"])
#     elif args.prompt_type == "jiuzhang_fs":
#         full_prompt = "## Question\n{instruction}\n\n## Solution\n"
#         full_prompt = full_prompt.format(instruction=example["question"])
#         full_prompt = f"{demo_prompt.strip()}\n\n\n{full_prompt}"
#     elif args.prompt_type == "mmiqc":
#         full_prompt = 'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{instruction}\n\n'
#         full_prompt = full_prompt.format(instruction=example["question"])
#     elif args.prompt_type == "abel":
#         full_prompt = "Question:\n{instruction}\nAnswer:\nLet's think step by step.\n"
#         full_prompt = full_prompt.format(instruction=example["question"])
#     elif args.prompt_type == "shepherd":
#         full_prompt = "{instruction}"
#         full_prompt = full_prompt.format(instruction=example["question"])

#     else:
#         raise NotImplementedError(args.prompt_type)
#     return full_prompt


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
