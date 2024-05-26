import asyncio
import multiprocessing
import os
import time
from tqdm import tqdm
from dataclasses import asdict, dataclass
import random
from vllm.utils import Counter as VllmCounter

from datasets import Dataset, load_dataset, load_from_disk, disable_caching

disable_caching()
from huggingface_hub import AsyncInferenceClient

from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, HfArgumentParser

from src.evol.openai_backend import LLM


@dataclass
class Args:
    # gneration parameters
    model: str = "Mistral"
    max_new_tokens: int = 2500
    """Max new tokens"""
    temperature: float = 0.6
    """Generation temperature"""
    top_p: float = 0.95
    """Generation top_p"""
    top_k: int = 50
    """Generation top_k"""
    repetition_penalty: float = 1.2
    """Generation repetition_penalty"""
    # prompts dataset parameters
    prompts_dataset: str = "HuggingFaceTB/cosmopedia-100k"
    """Dataset containing the prompts"""
    max_samples: int = 5000
    """The maximum number of samples to generate (use -1 for all))"""
    start_sample: int = -1
    """First sample to process"""
    end_sample: int = -1
    """Last sample to process"""
    seed: int = 42
    """Seed for shuffling"""
    prompt_column: str = "prompt"
    """Name of the column containing the prompt"""
    shuffle_dataset: bool = False
    """Whether to shuffle the prompts"""
    debug: bool = False
    """Debugging mode"""
    # logging parameters
    repo_id: str = "HuggingFaceTB/synthetic_data_test"
    """The repo id to push to"""
    checkpoint_path: str = "./synthetic_data"
    """Path for saving intermediate generations"""
    checkpoint_interval: int = 1_000
    """Interval for saving intermediate generations"""
    wandb_username: str = "zhangbeichen724"
    """Wandb username"""
    min_token_length: int = 150
    """Minimum number of tokens in a generation to be kept in the final dataset"""
    push_to_hub: bool = True
    """Whether to push to hub"""
    num_gpus: int = 5
    disable_async: bool = False
    api_key_path: str = "openai_api_key"


def process_text(examples, args):
    # apply chat template
    prompts = []
    for p in examples[args.prompt_column]:
        prompts.append([{"role": "user", "content": p}])
    return {"chat_prompt": prompts}


def get_predictions(outputs):
    predictions = [[] for _ in range(len(outputs))]
    new_token_lens = [[] for _ in range(len(outputs))]
    for output in outputs:
        predictions[int(output.request_id)] = output.outputs[0].text
        new_token_lens[int(output.request_id)] = len(output.outputs[0].token_ids)
    return predictions, new_token_lens


def process_batch(args, llm, prompts):
    if not args.disable_async:
        batch_outputs = asyncio.run(
            llm.achat(
                prompts,
                model=args.model,
                stop=["```output", "---"],
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_beams=1,
            )
        )
        return [generations[0] for generations in batch_outputs]
    else:
        batch_outputs = []
        for prompt in tqdm(prompts):
            batch_outputs.append(
                llm.chat(
                    prompt,
                    model=args.model,
                    stop=["```output", "---"],
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    num_beams=1,
                )
            )
        return [generations[0] for generations in batch_outputs]


def main():
    start_time = time.time()
    saving_time = 0

    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)

    ds = load_from_disk(args.prompts_dataset)
    print(ds)

    if args.shuffle_dataset:
        ds = ds.shuffle(seed=args.seed)

    if args.start_sample >= 0:
        end_sample = len(ds) if args.end_sample < 0 else args.end_sample
        print(
            f"Loading a defined range of samples: ({args.start_sample}, {end_sample})..."
        )
        ds = ds.select(range(args.start_sample, end_sample))
    elif args.max_samples > 0:
        print(f"Loading the first {args.max_samples} samples...")
        ds = ds.select(range(args.max_samples))

    repo_id = (
        f"{args.repo_id}_{args.prompt_column}"
        if args.prompt_column not in args.repo_id
        else args.repo_id
    )

    repo_id = (
        f"{args.repo_id}_{args.prompt_column}"
        if args.prompt_column not in args.repo_id
        else args.repo_id
    )
    checkpoint_dir = f"{args.checkpoint_path}/{repo_id.split('/')[1]}/data"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Will be saving at {checkpoint_dir}")
    ds = ds.map(
        process_text,
        batched=True,
        num_proc=16,
        fn_kwargs={"args": args},
    )

    for index in random.sample(range(len(ds)), min(3, len(ds))):
        print(f"Sample {index} of the test set: {ds[index]}.")

    total_samples = len(ds)
    ckpt_base_idx = 0 if args.start_sample < 0 else args.start_sample
    llm = LLM(api_key_path=args.api_key_path)
    for i in range(0, total_samples, args.checkpoint_interval):
        batch_time = time.time()
        # Processing a chunk
        print(
            f"Processing chunk {int(i/args.checkpoint_interval)}/{int(total_samples/args.checkpoint_interval)}"
        )
        end_index = min(i + args.checkpoint_interval, total_samples)
        chunk = ds.select(range(i, end_index))
        predictions = process_batch(args, llm, chunk["chat_prompt"])
        chunk = chunk.add_column("generated_text", predictions)
        # Save the chunk results and log throughput
        temp_time = time.time()
        time_per_chunk = temp_time - batch_time
        start_idx = ckpt_base_idx + i
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{start_idx}.json")
        chunk.to_json(checkpoint_path)
        saving_time += time.time() - temp_time
        print(
            f"💾 Checkpoint (samples {start_idx}-{start_idx + args.checkpoint_interval}) saved at {checkpoint_path}."
        )

    end_time = time.time()

    print(
        "Done processing and saving all chunks 🎉! Let's get some stats and push to hub..."
    )
    total_duration = end_time - start_time
    print(
        f"Total duration: {total_duration // 3600}h{int((total_duration % 3600) // 60)}min "
    )
    print(f"Saving time: {saving_time}s={saving_time/60}min ")


main()
# wandb.finish()
