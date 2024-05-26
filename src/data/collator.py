from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

from src.common.templates import IGNORE_INDEX


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def tensor_and_pad(self, tensor, pad_token_id, padding_side=None):
        padding_side = (
            self.tokenizer.padding_side if padding_side is None else padding_side
        )
        if padding_side == "left":
            tensor = [torch.tensor(x[::-1]) for x in tensor]
        else:
            tensor = [torch.tensor(x) for x in tensor]
        tensor = torch.nn.utils.rnn.pad_sequence(
            tensor, batch_first=True, padding_value=pad_token_id
        )
        if padding_side == "left":
            return tensor.flip(dims=[1])
        return tensor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids, labels = tuple(
        #     [instance[key] for instance in instances] for key in ("input_ids", "labels")
        # )
        # input_ids = [torch.tensor(x) for x in input_ids]
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        # )
        # labels = [torch.tensor(x) for x in labels]
        # labels = torch.nn.utils.rnn.pad_sequence(
        #     labels, batch_first=True, padding_value=IGNORE_INDEX
        # )
        # return dict(
        #     input_ids=input_ids,
        #     labels=labels,
        #     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        # )


        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = self.tensor_and_pad(input_ids, self.tokenizer.pad_token_id)
        features = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).to(
                dtype=input_ids.dtype
            ),
        )
        if "labels" in instances[0]:
            labels = [instance["labels"] for instance in instances]
            labels = self.tensor_and_pad(labels, IGNORE_INDEX)
            features["labels"] = labels
        return features


@dataclass
class DataCollatorForEval(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def tensor_and_pad(self, tensor, pad_token_id, padding_side=None):
        padding_side = (
            self.tokenizer.padding_side if padding_side is None else padding_side
        )
        if padding_side == "left":
            tensor = [torch.tensor(x[::-1]) for x in tensor]
        else:
            tensor = [torch.tensor(x) for x in tensor]
        tensor = torch.nn.utils.rnn.pad_sequence(
            tensor, batch_first=True, padding_value=pad_token_id
        )
        if padding_side == "left":
            return tensor.flip(dims=[1])
        return tensor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = self.tensor_and_pad(input_ids, self.tokenizer.pad_token_id)
        features = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).to(
                dtype=input_ids.dtype
            ),
        )
        if "labels" in instances[0]:
            labels = [instance["labels"] for instance in instances]
            labels = self.tensor_and_pad(labels, IGNORE_INDEX)
            features["labels"] = labels
        return features


@dataclass
class DataCollatorForSupervisedDatasetCringe(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        if isinstance(instances[0]["classifier_labels"], int):
            classifier_labels = torch.tensor(
                [instance["classifier_labels"] for instance in instances]
            )
        else:
            classifier_labels = [
                torch.tensor(instance["classifier_labels"]) for instance in instances
            ]
            classifier_labels = torch.nn.utils.rnn.pad_sequence(
                classifier_labels, batch_first=True, padding_value=1
            )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            classifier_labels=classifier_labels,
        )


@dataclass
class DataCollatorForSupervisedDatasetMLEAug(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        len(instances)
        len(instances[0]["input_ids"])
        is_train = isinstance(instances[0]["input_ids"][0], list)
        if is_train:
            input_ids, labels = tuple(
                [feature for instance in instances for feature in instance[key]]
                for key in ("input_ids", "labels")
            )
        else:
            input_ids, labels = tuple(
                [instance[key] for instance in instances]
                for key in ("input_ids", "labels")
            )
        len(input_ids)
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class DataCollatorForSupervisedDatasetCL(DataCollatorForSupervisedDatasetMLEAug):
    def get_features(self, instances):
        is_train = isinstance(instances[0]["input_ids"][0], list)
        if is_train:
            input_ids, labels = tuple(
                [feature for instance in instances for feature in instance[key]]
                for key in ("input_ids", "labels")
            )
        else:
            input_ids, labels = tuple(
                [instance[key] for instance in instances]
                for key in ("input_ids", "labels")
            )
        input_ids = [torch.tensor(x) for x in input_ids]
        labels = [torch.tensor(x) for x in labels]
        return input_ids, labels

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        model_keys = [key for key in instances[0] if key != "reward"]
        positives = [
            {key: i[key] for key in model_keys if "negative" not in key}
            for i in instances
        ]
        input_ids, labels = self.get_features(positives)
        pos_batch_size = len(input_ids)
        negatives = [
            {
                key.replace("negative_", ""): i[key]
                for key in model_keys
                if "negative" in key
            }
            for i in instances
        ]
        if len(negatives[0].keys()) > 0:
            neg_input_ids, neg_labels = self.get_features(negatives)
            input_ids = input_ids + neg_input_ids
            labels = labels + neg_labels
            neg_reward = [r for instance in instances for r in instance["reward"]]
            pos_reward = [1 for _ in range(pos_batch_size)]
            reward = torch.tensor(pos_reward + neg_reward)
        else:
            pos_reward = [1 for _ in range(pos_batch_size)]
            reward = torch.tensor(pos_reward)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            reward=reward,
        )


@dataclass
class DataCollatorForSupervisedDatasetQTrans(DataCollatorForSupervisedDataset):
    def tensor_and_pad(self, tensor, pad_token_id):
        tensor = [torch.tensor(x) for x in tensor]
        tensor = torch.nn.utils.rnn.pad_sequence(
            tensor, batch_first=True, padding_value=pad_token_id
        )
        return tensor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, question_ids, repeat_starts, repeat_ends = tuple(
            [instance[key] for instance in instances]
            for key in (
                "input_ids",
                "labels",
                "question_ids",
                "repeat_starts",
                "repeat_ends",
            )
        )
        input_ids = self.tensor_and_pad(input_ids, self.tokenizer.pad_token_id)
        labels = self.tensor_and_pad(labels, IGNORE_INDEX)
        question_ids = self.tensor_and_pad(question_ids, self.tokenizer.pad_token_id)
        repeat_starts = torch.tensor(repeat_starts)
        repeat_ends = torch.tensor(repeat_ends)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            question_ids=question_ids,
            repeat_starts=repeat_starts,
            repeat_ends=repeat_ends,
        )


@dataclass
class DataCollatorForQTransEval(DataCollatorForSupervisedDatasetQTrans):
    def tensor_and_pad(self, tensor, pad_token_id, padding_side=None):
        padding_side = (
            self.tokenizer.padding_side if padding_side is None else padding_side
        )
        if padding_side == "left":
            tensor = [torch.tensor(x[::-1]) for x in tensor]
        else:
            tensor = [torch.tensor(x) for x in tensor]
        tensor = torch.nn.utils.rnn.pad_sequence(
            tensor, batch_first=True, padding_value=pad_token_id
        )
        if padding_side == "left":
            return tensor.flip(dims=[1])
        return tensor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ret = {}
        input_ids = [instance["input_ids"] for instance in instances]
        input_lens = torch.tensor([len(i) for i in input_ids])
        ret["input_ids"] = self.tensor_and_pad(input_ids, self.tokenizer.pad_token_id)
        max_len = ret["input_ids"].shape[1]
        ret["attention_mask"] = ret["input_ids"].ne(self.tokenizer.pad_token_id)
        if "question_ids" in instances[0]:
            question_ids, repeat_starts, repeat_ends = tuple(
                [instance[key] for instance in instances]
                for key in (
                    "question_ids",
                    "repeat_starts",
                    "repeat_ends",
                )
            )
            ret["question_ids"] = self.tensor_and_pad(
                question_ids, self.tokenizer.pad_token_id, padding_side="right"
            )
            ret["repeat_starts"] = torch.tensor(repeat_starts)
            ret["repeat_ends"] = torch.tensor(repeat_ends)
            if self.tokenizer.padding_side == "left":
                ret["repeat_starts"] += max_len - input_lens
                ret["repeat_ends"] += max_len - input_lens
        return ret


@dataclass
class DataCollatorForSupervisedDatasetCls(DataCollatorForSupervisedDataset):
    def tensor_and_pad(self, tensor, pad_token_id):
        tensor = [torch.tensor(x) for x in tensor]
        tensor = torch.nn.utils.rnn.pad_sequence(
            tensor, batch_first=True, padding_value=pad_token_id
        )
        return tensor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, reward = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "reward")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        reward = torch.tensor(reward)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            reward=reward,
        )


@dataclass
class DataCollatorForSupervisedDatasetRank(DataCollatorForSupervisedDataset):
    def tensor_and_pad(self, tensor, pad_token_id):
        tensor = [torch.tensor(x) for x in tensor]
        tensor = torch.nn.utils.rnn.pad_sequence(
            tensor, batch_first=True, padding_value=pad_token_id
        )
        return tensor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert len(instances) == 1
        input_ids, labels, rewards = tuple(
            instances[0][key] for key in ("input_ids", "labels", "rewards")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        rewards = torch.tensor(rewards)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            rewards=rewards,
        )


@dataclass
class DataCollatorForSupervisedDatasetRankCom(DataCollatorForSupervisedDatasetRank):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        features = super().__call__(instances)
        strategy_mask, strategy_ids, strategy_rewards = tuple(
            instances[0][key]
            for key in ("strategy_mask", "strategy_ids", "strategy_rewards")
        )
        strategy_mask = [torch.tensor(x) for x in strategy_mask]
        sequence_length = features["input_ids"].shape[-1]
        padded_masks = []
        for mask in strategy_mask:
            padded_mask = torch.zeros(sequence_length, dtype=torch.long)
            padded_mask[: mask.shape[-1]] = mask
            padded_masks.append(padded_mask)
        features["strategy_mask"] = torch.stack(padded_masks)
        features["strategy_ids"] = torch.tensor(strategy_ids)
        features["strategy_rewards"] = torch.tensor(strategy_rewards)
        return features


@dataclass
class DataCollatorForSupervisedDatasetChoice(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, choice_labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "choice_labels")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        choice_labels = [torch.tensor(x) for x in choice_labels]
        choice_labels = torch.nn.utils.rnn.pad_sequence(
            choice_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            choice_labels=choice_labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class DataCollatorForSupervisedDatasetChoiceCls(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, input_mask, choice_rewards = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "input_mask", "choice_rewards")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_mask = [torch.tensor(x) for x in input_mask]
        sequence_length = input_ids.shape[-1]
        padded_masks = []
        for mask in input_mask:
            padded_mask = torch.zeros(sequence_length, dtype=torch.long)
            padded_mask[: mask.shape[-1]] = mask
            padded_masks.append(padded_mask)
        choice_rewards = torch.tensor(choice_rewards)
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_mask=torch.stack(padded_masks),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            choice_rewards=choice_rewards,
        )


@dataclass
class DataCollatorForSupervisedDatasetAttnForEval(DataCollatorForEval):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, q_token_mask, k_token_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "q_token_mask", "k_token_mask")
        )
        input_ids = self.tensor_and_pad(input_ids, self.tokenizer.pad_token_id)
        q_token_mask = self.tensor_and_pad(q_token_mask, 0)
        k_token_mask = self.tensor_and_pad(k_token_mask, 0)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # print(input_ids[0])
        # print(q_token_mask[0])
        # print(k_token_mask[0])
        features = dict(
            input_ids=input_ids,
            q_token_mask=q_token_mask,
            k_token_mask=k_token_mask,
            attention_mask=attention_mask,
        )
        if "labels" in instances[0]:
            labels = [instance["labels"] for instance in instances]
            labels = self.tensor_and_pad(labels, IGNORE_INDEX)
            features["labels"] = labels
        if "tgt_index" in instances[0]:
            if self.tokenizer.padding_side == "right":
                tgt_index = torch.tensor(
                    [instance["tgt_index"] for instance in instances]
                )
            else:
                tgt_index = torch.full(
                    (input_ids.shape[0],),
                    fill_value=input_ids.shape[1],
                    dtype=torch.long,
                )
            features["tgt_index"] = tgt_index
        return features


@dataclass
class DataCollatorForSupervisedDatasetAttn(DataCollatorForEval):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, q_token_mask, k_token_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "q_token_mask", "k_token_mask")
        )
        input_ids = self.tensor_and_pad(input_ids, self.tokenizer.pad_token_id)
        labels = self.tensor_and_pad(labels, IGNORE_INDEX)
        q_token_mask = self.tensor_and_pad(q_token_mask, 0)
        k_token_mask = self.tensor_and_pad(k_token_mask, 0)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        features = dict(
            input_ids=input_ids,
            labels=labels,
            q_token_mask=q_token_mask,
            k_token_mask=k_token_mask,
            attention_mask=attention_mask,
        )
        if "tgt_index" in instances[0]:
            if self.tokenizer.padding_side == "right":
                tgt_index = torch.tensor(
                    [instance["tgt_index"] for instance in instances]
                )
            else:
                tgt_index = torch.full(
                    (input_ids.shape[0],),
                    fill_value=input_ids.shape[1],
                    dtype=torch.long,
                )
            features["tgt_index"] = tgt_index
        return features



@dataclass
class DataCollatorForSupervisedDatasetGrad(DataCollatorForEval):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, token_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "token_mask")
        )
        input_ids = self.tensor_and_pad(input_ids, self.tokenizer.pad_token_id)
        labels = self.tensor_and_pad(labels, IGNORE_INDEX)
        token_mask = self.tensor_and_pad(token_mask, False)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        features = dict(
            input_ids=input_ids,
            labels=labels,
            token_mask=token_mask,
            attention_mask=attention_mask,
        )
        return features

@dataclass
class DataCollatorForSupervisedDatasetReasonGraph(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        (
            input_ids,
            labels,
            all_reference_tuples,
            all_step_tuples,
            amplify_factors,
        ) = tuple(
            [instance[key] for instance in instances]
            for key in (
                "input_ids",
                "labels",
                "all_reference_tuples",
                "all_step_tuples",
                "amplify_factor",
            )
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        max_seq_len = input_ids.shape[-1]
        k_token_mask = []
        for i in range(len(all_reference_tuples)):
            cur_token_mask = torch.zeros(max_seq_len, max_seq_len)
            for ref_tuples, step_tuple in zip(
                all_reference_tuples[i], all_step_tuples[i]
            ):
                for ref_tuple in ref_tuples:
                    cur_token_mask[
                        step_tuple[0] : step_tuple[1], ref_tuple[0] : ref_tuple[1]
                    ] = amplify_factors[i]
            k_token_mask.append(cur_token_mask)
        k_token_mask = torch.stack(k_token_mask)
        return dict(input_ids=input_ids, labels=labels, k_token_mask=k_token_mask)


@dataclass
class DataCollatorForSupervisedDatasetRepeat(DataCollatorForEval):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "attention_mask")
        )
        input_ids = self.tensor_and_pad(input_ids, self.tokenizer.pad_token_id)
        attention_mask = self.tensor_and_pad(attention_mask, 0)
        features = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return features
