from vllm import LLM, SamplingParams
from datasets import load_from_disk
from transformers import AutoTokenizer
import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--start_sample", type=int, default=-1)
    parser.add_argument("--end_sample", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    ds = load_from_disk(args.data_path)
    try:
        ds = ds["train"]
    except:
        pass
    print(len(ds))

    if args.start_sample >= 0:
        end_sample = len(ds) if args.end_sample < 0 else args.end_sample
        print(
            f"Loading a defined range of samples: ({args.start_sample}, {end_sample})..."
        )
        ds = ds.select(range(args.start_sample, end_sample))
    elif args.max_samples > 0:
        print(f"Loading the first {args.max_samples} samples...")
        ds = ds.select(range(args.max_samples))

    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=5000)

    llm = LLM(model=args.model_path, tensor_parallel_size=1)
    outputs = llm.generate(ds["text"], sampling_params)

    generated_texts = []
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: \n{prompt}\n\nGenerated text: \n{generated_text}\n*****************************\n\n")
        generated_texts.append({"prompt": prompt, "generated_text": generated_text})
    with open(
        args.save_path,
        "w",
    ) as f:
        for text in generated_texts:
            f.write(json.dumps(text) + "\n")


if __name__ == "__main__":
    main()
