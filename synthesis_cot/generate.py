import asyncio
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
import random
from vllm.utils import Counter as VllmCounter

from datasets import Dataset, load_dataset, load_from_disk, disable_caching

disable_caching()
from huggingface_hub import AsyncInferenceClient

# from llm_swarm import LLMSwarm, LLMSwarmConfig
from vllm import SamplingParams, LLM
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, HfArgumentParser

# import wandb


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
    wandb_username: str = ""
    """Wandb username"""
    min_token_length: int = 150
    """Minimum number of tokens in a generation to be kept in the final dataset"""
    push_to_hub: bool = True
    """Whether to push to hub"""
    num_gpus: int = 5


def process_text(examples, args, tokenizer):
    # apply chat template
    prompts = []
    for p in examples[args.prompt_column]:
        if "Mixtral" in args.model:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
            )
        else:
            prompt = p
        prompts.append(prompt)
    return {"chat_prompt": prompts}


def get_predictions(outputs):
    predictions = [[] for _ in range(len(outputs))]
    new_token_lens = [[] for _ in range(len(outputs))]
    for output in outputs:
        predictions[int(output.request_id)] = output.outputs[0].text
        new_token_lens[int(output.request_id)] = len(output.outputs[0].token_ids)
    return predictions, new_token_lens


def main():
    start_time = time.time()
    total_tokens = 0
    saving_time = 0

    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    args_dict = asdict(args)
    # args used in wandb
    # args_dict = asdict(args)
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ds = load_from_disk(args.prompts_dataset)

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

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop="<|endoftext|>",
        stop_token_ids=[128009] if "Llama-3" in args.model else None,
    )
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        download_dir=None,
        tensor_parallel_size=args.num_gpus,
    )

    repo_id = (
        f"{args.repo_id}_{args.prompt_column}"
        if args.prompt_column not in args.repo_id
        else args.repo_id
    )
    # wandb.init(
    #     project="synthetic_data",
    #     entity=args.wandb_username,
    #     name=repo_id.split("/")[1],
    # )
    # # Convert args to dict
    # wandb.config.update(args_dict)

    repo_id = (
        f"{args.repo_id}_{args.prompt_column}"
        if args.prompt_column not in args.repo_id
        else args.repo_id
    )
    checkpoint_dir = f"{args.checkpoint_path}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Will be saving at {checkpoint_dir}")
    ds = ds.map(
        process_text,
        batched=True,
        num_proc=16,
        fn_kwargs={"tokenizer": tokenizer, "args": args},
    )

    for index in random.sample(range(len(ds)), min(3, len(ds))):
        print(f"Sample {index} of the test set: {ds[index]}.")

    total_samples = len(ds)
    ckpt_base_idx = 0 if args.start_sample < 0 else args.start_sample
    for i in range(0, total_samples, args.checkpoint_interval):
        batch_time = time.time()
        # Processing a chunk
        print(
            f"Processing chunk {int(i/args.checkpoint_interval)}/{int(total_samples/args.checkpoint_interval)}"
        )
        end_index = min(i + args.checkpoint_interval, total_samples)
        chunk = ds.select(range(i, end_index))
        # chunk_results = chunk.map(process_text, batched=True, num_proc=96)
        llm.request_counter = VllmCounter()
        try:
            outputs = llm.generate(chunk["chat_prompt"], sampling_params)
        except Exception as e:
            print("Error,", e)
            continue
        predictions, new_token_lens = get_predictions(outputs)
        predictions = predictions[-len(chunk) :]
        try:
            chunk = chunk.add_column("generated_text", predictions)
        except Exception as e:
            print("Error", e)
            continue
        # Save the chunk results and log throughput
        temp_time = time.time()
        time_per_chunk = temp_time - batch_time
        start_idx = ckpt_base_idx + i
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{start_idx}.json")
        # intermediate_ds = Dataset.from_list(chunk_results)
        chunk.to_json(checkpoint_path)
        batch_tokens = sum(new_token_lens)
        total_tokens += batch_tokens
        saving_time += time.time() - temp_time
        print(
            f"💾 Checkpoint (samples {start_idx}-{start_idx + args.checkpoint_interval}) saved at {checkpoint_path}."
        )
        # wandb.log(
        #     {
        #         "sample": i + args.checkpoint_interval,
        #         "batch": int(i / args.checkpoint_interval),
        #         "total_tokens (M)": total_tokens / 1e6,
        #         "tokens_per_batch": batch_tokens,
        #         "time_per_batch (s)": time_per_chunk,
        #         "generated_tokens_per_sec": int(batch_tokens / time_per_chunk),
        #     }
        # )

    end_time = time.time()

    print(
        "Done processing and saving all chunks 🎉! Let's get some stats and push to hub..."
    )
    total_duration = end_time - start_time
    overall_tokens_per_second = (
        total_tokens / total_duration if total_duration > 0 else 0
    )
    print(
        f"🏎️💨 Overall Tokens per Second: {overall_tokens_per_second:.2f}, per instance: {overall_tokens_per_second/len(ds):.2f}"
    )
    print(f"Generated {total_tokens / 1e6:.2f}M tokens")
    print(
        f"Total duration: {total_duration // 3600}h{int((total_duration % 3600) // 60)}min "
    )
    print(f"Saving time: {saving_time}s={saving_time/60}min ")

    # load dataset
    # print("Load checkpoints...")
    # output_ds = load_dataset(checkpoint_dir, split="train")
    # # remove empty completions
    # final_data = output_ds.filter(
    #     lambda x: x["token_length"] >= args.min_token_length
    # )
    # print(final_data)
    # failed = output_ds.filter(lambda x: x["token_length"] <= args.min_token_length)
    # print(final_data)
    # if args.push_to_hub:
    #     print(f"📨 Pushing dataset to {repo_id}")
    #     final_data.push_to_hub(repo_id, private=True)
    #     print("Dataset pushed!")
    #     if len(failed) > 0:
    #         print(f"{len(failed)} generations failed")
    #         size = min(len(failed), 1000)
    #         failed = failed.select(range(size))
    #         failed.push_to_hub(f"{repo_id}_failed", private=True)


main()
# wandb.finish()
