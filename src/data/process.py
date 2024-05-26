import re
import copy
from typing import Dict, Sequence

import torch
import transformers

IGNORE_INDEX = -100


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    return_offsets_mapping=False,
    add_eos_token=True,
    add_special_tokens=True,
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        for text in strings
    ]
    print("tokenizer.model_max_length", tokenizer.model_max_length)
    input_ids = labels = [
        (
            torch.cat([tokenized.input_ids[0], torch.tensor([tokenizer.eos_token_id])])
            if add_eos_token
            else tokenized.input_ids[0]
        )
        for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    attention_mask = [
        torch.cat([tokenized.attention_mask[0], torch.tensor([1])])
        for tokenized in tokenized_list
    ]
    ret = dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        attention_mask=attention_mask,
    )
    if return_offsets_mapping:
        offset_mapping = [
            tokenized["offset_mapping"][0].tolist() for tokenized in tokenized_list
        ]
        ret["offset_mapping"] = offset_mapping
    return ret


def extract_solution(target):
    if "solution()" in target:
        return target.split('"""')[-1]
    if "The answer is" in target:
        pattern = r"^>.*?$"
        return re.sub(pattern, "", target, flags=re.MULTILINE)
    return target


def find_sub_list(l, sl):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            return ind


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    model_args,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    solutions = [extract_solution(t) for t in targets]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    sources_tokenized = _tokenize_fn(sources, tokenizer, add_eos_token=False)
    solution_tokenized = _tokenize_fn(solutions, tokenizer, add_special_tokens=False)
    source_lens = sources_tokenized["input_ids_lens"]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    if data_args.apply_partial_tgt_loss:
        for label, input_id, target_id in zip(
            labels, input_ids, solution_tokenized["input_ids"]
        ):
            try:
                ts_idx = find_sub_list(input_id.tolist(), target_id[1:].tolist())
            except:
                print("Warning", label)
                ts_idx = label.shape[-1] - 1
            label[:ts_idx] = IGNORE_INDEX
    elif not data_args.apply_src_loss:
        for i, (label, source_len) in enumerate(zip(labels, source_lens)):
            if source_len <= 1:
                source_len = 0
            # assert source_len + len(solution_tokenized["input_ids"]) == len(label), (
            #     source_len,
            #     len(solution_tokenized["input_ids"]),
            #     len(label),
            # )
            label[:source_len] = IGNORE_INDEX
    # if model_args.copy_question:
    if data_args.copy_question:
        assert data_args.apply_src_loss
        for i in range(len(input_ids)):
            labels[i] = torch.cat(
                [labels[i][: source_lens[i] - 1], labels[i][source_lens[i] :]], dim=0
            )
            input_ids[i] = input_ids[i][:-1]
    return dict(input_ids=input_ids, labels=labels)


def process_dataset(examples, tokenizer, data_args, model_args):
    data_dict = preprocess(
        examples["input"], examples["output"], tokenizer, data_args, model_args
    )
    return data_dict
