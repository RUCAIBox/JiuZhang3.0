import numpy as np
from typing import Dict, Union
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import transformers
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_callback import TrainerCallback

from typing import Optional

import datasets
import torch
from torch.utils.data import DataLoader, sampler, SequentialSampler
from transformers import Trainer, StoppingCriteria, LogitsProcessor
from transformers.trainer_pt_utils import IterableDatasetShard, LengthGroupedSampler
from transformers.trainer_utils import seed_worker, has_length
from transformers.utils import is_datasets_available


class GumbelTempCallback(TrainerCallback):
    def get_gumbel_temperature(self, args, global_step, max_steps):
        return np.maximum(
            # args.initial_tau * np.exp(-args.anneal_rate * steps),
            args.trans_initial_tau * np.exp(-global_step / max_steps),
            args.trans_minimum_tau,
        )

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        tau = self.get_gumbel_temperature(args, state.global_step, state.max_steps)
        model = kwargs["model"]
        model.set_gumbel_temperature(tau)
        if dist.get_rank() == 0:
            print("tau", tau)


class SoftThresholding(nn.Module):
    def __init__(self, a_init, b_init, k):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([a_init]), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([b_init]), requires_grad=True)
        self.k = k

    def forward(self, x, mean, std):
        a = torch.tanh(self.a)
        b = torch.tanh(self.b)
        a_threshold = mean + a * std
        b_threshold = mean + b * std
        return torch.sigmoid(self.k * (x - a_threshold)) * torch.sigmoid(
            self.k * (b_threshold - x)
        )


class SoftThresholdingCenter(nn.Module):
    def __init__(
        self, center_init, half_width_init, k, num_heads=None, activation=False
    ):
        super(SoftThresholdingCenter, self).__init__()
        if num_heads is None:
            self.center = nn.Parameter(torch.tensor([center_init]))
            self.half_width = nn.Parameter(torch.tensor([half_width_init]))
        else:
            self.center = nn.Parameter(torch.tensor([center_init] * num_heads))
            self.half_width = nn.Parameter(torch.tensor([half_width_init] * num_heads))
        self.num_heads = num_heads
        self.activation = activation
        self.k = k

    def forward(self, s, mean, std):
        if self.activation:
            center = torch.tanh(self.center)
            half_width = torch.sigmoid(self.half_width)  # Ensure half_width is positive
        else:
            center = torch.clamp(self.center, -0.5, 0.5)
            half_width = torch.clamp(self.half_width, 0.0, 0.5)
        a = center - half_width
        b = center + half_width
        # print("center, half_width", self.center, self.half_width)
        if self.num_heads is None:
            a_threshold = mean + a * std
            b_threshold = mean + b * std
        else:
            a_threshold = mean + a[None, :, None, None] * std
            b_threshold = mean + b[None, :, None, None] * std
        soft_mask = torch.sigmoid(self.k * (s - a_threshold)) * (
            1 - torch.sigmoid(self.k * (s - b_threshold))
        )
        # if dist.is_initialized() and dist.get_rank() == 0:
        #     print("ab factor", a, b)
        #     #     # print("a_threshold", a_threshold[0, 0])
        #     #     # print("b_threshold", b_threshold[0, 0])
        #     #     # print("weight", s[0, 0])
        #     print("delta", self.k * (s - a_threshold)[0, 0].squeeze())
        #     #     print("self_k", self.k)
        #     print(
        #         "delta_sigmoid",
        #         torch.sigmoid(self.k * (s - a_threshold))[0, 0].squeeze(),
        #     )
        #     if soft_mask.shape[2] > 20:
        #         print("soft_mask", soft_mask[0, 0, -20])
        # #     print("soft_mask_max", torch.max(soft_mask[0, 0]))
        # #     print("soft_mask_min", torch.min(soft_mask[0, 0]))
        return soft_mask


def _nearest_divisible(num: int, divisor: int) -> int:
    """Returns the nearest number to `num` that is divisible by `divisor`."""
    return (num + divisor - 1) // divisor * divisor


def smart_tokenizer_and_embedding_resize(
    special_tokens: Dict[str, Union[str, transformers.AddedToken]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
    is_special: bool = False,
):
    if is_special:
        num_new_tokens = tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
    else:
        num_new_tokens = tokenizer.add_tokens(special_tokens)
    old_size = len(tokenizer)
    new_size = _nearest_divisible(num=old_size + num_new_tokens, divisor=64)
    model.resize_token_embeddings(new_size)
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class NoShuffleSeq2SeqTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = SequentialSampler(self.train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["shuffle"] = False

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


def get_wsd_scheduler(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, stable_ratio=0.9
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        num_stable_steps = (
            stable_ratio * num_training_steps
            if stable_ratio != 0.0
            else num_warmup_steps
        )
        if current_step < num_stable_steps:
            return 1.0
        return max(
            0.1,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_stable_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class WSDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_wsd_scheduler(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                stable_ratio=self.args.stable_ratio,
            )
            self._created_lr_scheduler = True
            print("Use get wsd")
        return self.lr_scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        optimizer = self.optimizer
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=optimizer
        )
        print("Scheduler", self.lr_scheduler)


class WSDNoShuffleTrainer(NoShuffleSeq2SeqTrainer, WSDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], eos_token_id=0):
        super().__init__()
        self.stops = stops
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for seq in input_ids:
            for stop in self.stops:
                if (
                    len(seq) >= len(stop)
                    and torch.all((stop == seq[-len(stop) :])).item()
                ):
                    seq[-len(stop)] = self.eos_token_id
                    # return True
        return False


class LogitsProcessorSub(LogitsProcessor):
    def __init__(self, stops=[], eos_token_id=0):
        super().__init__()
        self.stops = stops
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        forced_eos = torch.full((scores.size(1),), -torch.inf, device=scores.device)
        forced_eos[self.eos_token_id] = 0
        for stop in self.stops:
            if input_ids.shape[1] < len(stop):
                continue
            scores[torch.all((stop == input_ids[:, -stop.shape[-1] :]), dim=-1)] = (
                forced_eos
            )
        return scores


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        super().__init__()
        assert isinstance(
            stop_id_sequences[0], list
        ), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence) :].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


class WSDSaveModelCallback(TrainerCallback):
    def __init__(self, save_percentage=0.9):
        self.save_percentage = save_percentage
        self.saved = False

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        total_steps = state.max_steps

        if not self.saved and current_step >= self.save_percentage * total_steps:
            control.should_save = True
            self.saved = True
