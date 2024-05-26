import datetime
import json
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch.utils.data import Dataset



def merge_data_points_by_length(lengths: List[int], max_length: int) -> List[List[int]]:
    """given lengths of data points, we merge them into groups such that the sum of lengths
    in each group is less than max_length. This is known as: https://en.wikipedia.org/wiki/Bin_packing_problem
    Here is the greedy algorithm
    Args:
        lengths (List[int]): _description_
        max_length (int): _description_

    Returns:
        _type_: groups of indices: [[index1, index2, ...], [], ...]
    """
    items = [{"length": length, "index": i} for i, length in enumerate(lengths)]
    items = sorted(items, key=lambda x: x["index"])
    merges = []
    current_sum = 0
    current_list = []
    for i in range(len(items)):
        cur_length = items[i]["length"]
        if cur_length + current_sum <= max_length:
            current_sum += items[i]["length"]
            current_list.append(i)
        else:
            merges.append(current_list)
            current_list = [i]
            current_sum = cur_length

    if len(current_list) > 0:
        merges.append(current_list)

    result = []
    for merge in merges:
        sub_items = [items[index]["index"] for index in merge]
        result.append(sub_items)
    return result


def get_causal_mask(length: int, m_value: float) -> torch.tensor:
    """Return causal mask filling with m_value

    Args:
        length (int): _description_
        m_value (float): _description_

    Returns:
        torch.tensor: _description_
    """
    mask = torch.full((length, length), m_value)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    return mask


def create_mask_from_lengths(
    lengths: List[int], pack_length: int, m_value: float
) -> torch.tensor:
    """create attention_mask: N x N where masked value = m_value
    Args:
        lengths (List[int]): length of data points
        tokenizer (Any): _description_
        m_value (float): _description_

    Returns:
        torch.tensor: _description_
    """
    max_length = pack_length
    result = torch.full((max_length, max_length), m_value)
    acc_leng = 0
    for length in lengths:
        # mask for a data point with length
        x = get_causal_mask(length, m_value)
        result[acc_leng : acc_leng + length, acc_leng : acc_leng + length] = x
        acc_leng += length

    pad_length = max_length - sum(lengths)
    if pad_length > 0:
        result[-pad_length:, :] = 0
        result[:, -pad_length:] = m_value
    return result


def pack_data_points(data_points: List[Dict], tokenizer: Any, pack_length: int) -> Dict:
    """This method is used to pack multiple data points into a single data point used for Normal Attention (vs FlashAttention)

    Args:
        data_points (List[Dict]): _description_
        tokenizer (Any): _description_

    Returns:
        Dict: _description_
    """
    input_ids = []
    lengths = []
    label_ids = []
    for item in data_points:
        input_ids += item["input_ids"]
        # assert item["labels"][0] == -100 # This is to make sure that the first token won't be included in computing loss
        labels = list(item["labels"])
        labels[0] = -100
        label_ids += labels
        lengths.append(len(item["input_ids"]))

    attention_mask = create_mask_from_lengths(lengths, pack_length, float("-inf"))
    pad_leng = pack_length - len(input_ids)  # padding to model_max_length

    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
        label_ids = label_ids + [-100 for _ in range(pad_leng)]
    else:
        input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
        label_ids = [-100 for _ in range(pad_leng)] + label_ids

    assert len(input_ids) == len(label_ids) == attention_mask.size(0) == pack_length

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.unsqueeze(
            attention_mask, 0
        ),  # unsqueeze <-- because the shape is: B x 1 x N x N
    }


def pack_data_points_FA(
    data_points: List[Dict], tokenizer: Any, pack_length: int
) -> Dict:
    """This method is used to pack multiple data_points into a single data point usable for Flash Attention

    For example, we want to pack 2 inputs with padding_size=right:
    input1= {"input_ids": token_ids1, "labels": label_ids1}
    input2= {"input_ids": token_ids2, "labels": label_ids2}
    --> output would be:

    output = {"input_ids": token_ids1 + token_ids + [pad_token, ...]} padding to tokenizer.model_max_length
    output["labels"] =  label_ids1 + label_ids2 + [-100, -100, ...]
    output["attention_mask"] = [1,...,1, 2,...,2, 0...0]
        number of 1s = len(input_ids1)
        number of 2s = len(input_ids2)
        number of 0s = padding_length

    Args:
        data_points (List[Dict]): List of data points to pack: [{"input_ids": xxx, "labels": xxx}, ...]
        tokenizer (Any): _description_

    Returns:
        Dict: final single data point
    """
    input_ids = []
    lengths = []
    label_ids = []
    attention_mask = []

    for index, item in enumerate(data_points):
        input_ids += item["input_ids"]
        # assert item["labels"][0] == -100 # This is to make sure that the first token won't be included in computing loss
        labels = list(item["labels"])
        labels[0] = -100
        label_ids += labels
        lengths.append(len(item["input_ids"]))
        attention_mask += [index + 1 for _ in range(len(item["input_ids"]))]

    pad_leng = pack_length - len(input_ids)  # padding to model_max_length

    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
        label_ids = label_ids + [-100 for _ in range(pad_leng)]
        attention_mask = attention_mask + [0 for _ in range(pad_leng)]
    else:
        input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
        label_ids = [-100 for _ in range(pad_leng)] + label_ids
        attention_mask = [0 for _ in range(pad_leng)] + attention_mask

    assert len(input_ids) == len(label_ids) == len(attention_mask) == pack_length
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.tensor(
            attention_mask
        ),  # unsqueeze <-- because the shape is: B x 1 x N x N
    }

class CachedDataset(Dataset):
    """This class implements a dataset that can be cached in a folder

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self, tokenizer: Any, cached_folder: str, ignore_cached: bool = False
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_points = []
        self.load_from_cache = False
        if cached_folder is not None and not ignore_cached:
            data_path = self.get_data_point_path(cached_folder)
            if os.path.exists(data_path):
                print(f"cached found, load from cached: {cached_folder}")
                self.load(cached_folder)
                self.load_from_cache = True

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data_points[i]

    def create_meta_info(self):
        return {
            "max_length": self.tokenizer.model_max_length,
            "size": len(self.data_points),
        }

    def load(self, folder: str):
        t1 = datetime.datetime.now()
        with open(self.get_data_point_path(folder), "rb") as file:
            self.data_points = pickle.load(file)
        t2 = datetime.datetime.now()
        print("time for loading cached data: ", (t2 - t1).total_seconds())

    def get_data_point_path(self, folder: str) -> str:
        return os.path.join(folder, "data_points.pkl")

    def get_metainfo_path(self, folder: str) -> str:
        return os.path.join(folder, "meta_info.json")

    def dump(self, folder: str):
        t1 = datetime.datetime.now()
        if not os.path.exists(folder):
            os.mkdir(folder)

        print("Save", len(self.data_points))

        with open(self.get_data_point_path(folder), "wb") as file:
            pickle.dump(self.data_points, file)

        with open(self.get_metainfo_path(folder), "w") as f:
            f.write(json.dumps(self.create_meta_info()))
        t2 = datetime.datetime.now()
        print("time for dumping data: ", (t2 - t1).total_seconds())

    def stat(self):
        print(json.dumps(self.create_meta_info()))


class PackedDataset(CachedDataset):
    """This class is used for Packing without Flash Attention"""

    def __init__(
        self,
        dataset: List[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        cached_folder: Optional[str] = None,
        ignore_cached: bool = False,
        batch_size: int = 5000,
        keep_assistant_prefix: bool = False,
        use_flash_attention: bool = True,
        pack_length: Optional[int] = None,
    ):
        super().__init__(tokenizer, cached_folder, ignore_cached)
        self.use_flash_attention = use_flash_attention
        self.pack_length = pack_length if pack_length else tokenizer.model_max_length
        print("self.pack_length: ", self.pack_length)
        if not self.load_from_cache:
            self.data_points = dataset
            self.update_packing_info()
            if cached_folder is not None:
                print(f"dump data to cached: {cached_folder}")
                self.dump(cached_folder)
        else:
            self.update_packing_info()

    def update_packing_info(self):
        self.lengths = self.data_points.map(lambda x: {"length": len(x)}, load_from_cache_file=False, num_proc=32, input_columns="input_ids")["length"]
        self.groups = merge_data_points_by_length(self.lengths, self.pack_length)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        group = self.groups[i]
        group_data_points = [self.data_points[index] for index in group]
        if not self.use_flash_attention:
            return pack_data_points(group_data_points, self.tokenizer, self.pack_length)
        return pack_data_points_FA(group_data_points, self.tokenizer, self.pack_length)

    def stat(self):
        print(
            f"number of original data points:{len(self.data_points)}; packed to: {len(self.groups)} data points"
        )
        original_avg_length = sum(self.lengths) / len(self.lengths)
        packed_lengths = []
        for group in self.groups:
            lengths = [self.lengths[index] for index in group]
            packed_lengths.append(sum(lengths))
        avg_packed_length = sum(packed_lengths) / len(packed_lengths)
        print(
            f"original avg length: {original_avg_length}; avg packed length: {avg_packed_length}"
        )