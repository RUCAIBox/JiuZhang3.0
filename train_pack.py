import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import random

import torch
import torch.distributed as dist
import transformers
import datasets
from transformers import (
    LlamaForCausalLM,
    Trainer,
    set_seed,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import disable_caching

disable_caching()


from src.utils.train_utils import (
    NoShuffleSeq2SeqTrainer,
    WSDTrainer,
    WSDNoShuffleTrainer,
    WSDSaveModelCallback,
)
from src.data.process import process_dataset

from src.packing.custom_dataset import PackedDataset
from src.packing.monkey_patch_packing import (
    monkey_patch_packing_mistral,
    monkey_patch_packing_mixtral,
)

monkey_patch_packing_mistral()
monkey_patch_packing_mixtral()
from src.packing.monkey_patch_packing import monkey_patch_packing_llama

monkey_patch_packing_llama()



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attention: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    no_shuffle: bool = field(
        default=False, metadata={"help": "Whether to shuffle the training data."}
    )
    reverse_order: bool = field(
        default=False, metadata={"help": "Whether to reverse the order of the data."}
    )
    max_samples: int = field(
        default=None, metadata={"help": "Maximum number of samples to use."}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    apply_src_loss: bool = field(default=False)
    apply_partial_tgt_loss: bool = field(default=False)
    copy_question: bool = field(default=False)
    prompt_format: str = field(default=None)
    only_output: bool = field(default=False)
    pack_cached_folder: str = field(default=None)
    disable_pack: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_wsd: bool = field(default=False)
    stable_ratio: float = field(default=0.9)


def make_supervised_data_module(tokenizer, data_args, training_args, model_args):
    if os.path.exists(os.path.join(data_args.data_path, "dataset_dict.json")):
        # if "gsm" in data_args.data_path and "gsm8k_cringe" not in data_args.data_path:
        raw_train_dataset = datasets.DatasetDict.load_from_disk(data_args.data_path)[
            "train"
        ]
    else:
        raw_train_dataset = datasets.Dataset.load_from_disk(data_args.data_path)
    # train_dataset = train_dataset.select(range(10))
    if data_args.max_samples is not None:
        raw_train_dataset = raw_train_dataset.select(range(data_args.max_samples))

    fn_kwargs = {
        "tokenizer": tokenizer,
        "data_args": data_args,
        "model_args": model_args,
    }

    if hasattr(training_args, "local_rank") and training_args.local_rank == 0:
        for index in range(5):
            print(f"Sample {index} of the training set: {raw_train_dataset[index]}.")

    if data_args.pack_cached_folder:
        cached_folder = data_args.pack_cached_folder
    else:
        cached_folder = os.path.join(training_args.output_dir, f"cached")

    if hasattr(training_args, "local_rank") and training_args.local_rank > 0:
        print(
            f"process: {training_args.local_rank} wait for main process to prepare the training data"
        )
        torch.distributed.barrier()
    else:
        if "input_ids" not in raw_train_dataset.column_names:
            train_dataset = raw_train_dataset.map(
                process_dataset,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_train_dataset.column_names,
                desc="Running tokenizer on train dataset",
                fn_kwargs=fn_kwargs,
            )
        else:
            train_dataset = raw_train_dataset
        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)
        if not os.path.exists(cached_folder):
            os.mkdir(cached_folder)
        print(f"train size: : {len(raw_train_dataset)}")
        train_dataset = PackedDataset(
            train_dataset,
            tokenizer,
            cached_folder=cached_folder,
            ignore_cached=True,
            use_flash_attention=True,
            pack_length=training_args.model_max_length + 1,
        )
        print(f"process: {training_args.local_rank} finish processing data")
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            print(world_size)
            torch.distributed.barrier()  # allow other ranks to execute

    train_dataset = PackedDataset(
        None,
        tokenizer,
        cached_folder=cached_folder,
        ignore_cached=False,
        use_flash_attention=True,
        pack_length=training_args.model_max_length + 1,
    )
    if training_args.local_rank == 0:
        train_dataset.stat()

    print(len(train_dataset))
    if hasattr(training_args, "local_rank") and training_args.local_rank == 0:
        for index in [0] + list(random.sample(range(len(train_dataset)), 3)):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")
            if isinstance(train_dataset[index]["input_ids"][0], list):
                print(tokenizer.decode(train_dataset[index]["input_ids"][0]))
            else:
                print(tokenizer.decode(train_dataset[index]["input_ids"]))
    return dict(train_dataset=train_dataset)


def get_model_tokenizer(model_args, data_args, training_args):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2" if model_args.flash_attention else None,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if "llama-3" in model_args.model_name_or_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token

    return model, tokenizer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = get_model_tokenizer(model_args, data_args, training_args)
    set_seed(training_args.seed)
    random.seed(training_args.seed)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        model_args=model_args,
    )
    model.is_parallelizable = True
    model.model_parallel = True
    trainer_class = Trainer
    if data_args.no_shuffle:
        if training_args.use_wsd:
            trainer_class = WSDNoShuffleTrainer
        else:
            trainer_class = NoShuffleSeq2SeqTrainer
    elif training_args.use_wsd:
        trainer_class = WSDTrainer
    save_model_callback = WSDSaveModelCallback(
        save_percentage=training_args.stable_ratio
    )
    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # TODO
        # callbacks=[save_model_callback],
        **data_module,
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
