import torch
import tqdm
import json
import time
import asyncio
import os
from transformers import StoppingCriteria
import numpy as np
import string

from src.utils.data_utils import extract_cot_answer, extract_answer_number, get_subtemp
from src.utils.code_utils import extract_code_answer, GenericRuntime
from src.utils.mwp_utils import extract_infix_answer, extract_prefix_answer
from src.common.templates import INFERENCE_PATTERN, PROMPT_DICT


extract_answer_fn = {
    "gsm8k": extract_answer_number,
    "gsm8k_question": extract_answer_number,
    "mwp_infix": extract_infix_answer,
    "mwp_prefix": extract_prefix_answer,
    "pal": extract_code_answer,
    "pal_question": extract_code_answer,
    "pal_plain": extract_code_answer,
}


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(
            stop_id_sequences[0], list
        ), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence) :].tolist() == stop_sequence:
                    sequences_should_be_stopped.append(True)
                    break
            sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    disable_tqdm=False,
    **generation_kwargs,
):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)]
                if stop_id_sequences
                else None,
                **generation_kwargs,
            )

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(
                        batch_input_ids.shape[1], batch_outputs.shape[1]
                    ):
                        if any(
                            batch_outputs[
                                output_idx, token_idx : token_idx + len(stop_sequence)
                            ].tolist()
                            == stop_sequence
                            for stop_sequence in stop_id_sequences
                        ):
                            batch_outputs[
                                output_idx, token_idx:
                            ] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True
            )
            batch_prompts = tokenizer.batch_decode(
                batch_input_ids, skip_special_tokens=True
            )
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [
                prompt for prompt in batch_prompts for _ in range(num_return_sequences)
            ]
            batch_generations = [
                output[len(prompt) :]
                for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences)

    assert (
        len(generations) == len(prompts) * num_return_sequences
    ), "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


@torch.no_grad()
def get_next_word_predictions(
    model,
    tokenizer,
    prompts,
    candidate_token_ids=None,
    batch_size=1,
    return_token_predictions=False,
    disable_tqdm=False,
):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_logits = model(batch_input_ids, attention_mask).logits[:, -1, :]
        if candidate_token_ids is not None:
            batch_logits = batch_logits[:, candidate_token_ids]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [
                    candidate_tokens[idx] for idx in batch_prediction_indices
                ]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(
                    batch_prediction_indices
                )
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(
        prompts
    ), "number of predictions should be equal to number of prompts"
    return predictions, probs


@torch.no_grad()
def score_completions(model, tokenizer, scoring_examples, disable_tqdm=False):
    """
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    """

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(scoring_examples), desc="Scoring Completions")

    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({"prompt": prompt, "completion": completion})

    scores = []
    # currently we don't support batching, because we want to directly use the loss returned by the model to score each completion.
    for unrolled_example in unrolled_examples:
        encoded_example = encode_with_prompt_completion_format(
            unrolled_example, tokenizer, max_seq_length=None
        )
        # unsqueeze the batch dimension
        for key, value in encoded_example.items():
            encoded_example[key] = value.unsqueeze(0)
        if model.device.type == "cuda":
            encoded_example = {
                key: value.cuda() for key, value in encoded_example.items()
            }
        outputs = model(**encoded_example)
        loss = outputs.loss
        scores.append(-loss.item())
        if not disable_tqdm:
            progress.update(1)

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores


def load_hf_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    load_in_8bit=False,
    load_in_half=False,
    gptq_model=False,
    use_fast_tokenizer=False,
    padding_side="left",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, use_fast=use_fast_tokenizer
    )
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM

        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device_map, load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map=device_map
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            if torch.cuda.is_available():
                model = model.cuda()
        if load_in_half:
            model = model.half()
    model.eval()
    return model, tokenizer


def query_openai_chat_model(
    engine,
    instances,
    output_path=None,
    batch_size=10,
    retry_limit=5,
    reuse_existing_outputs=True,
    **completion_kwargs,
):
    """
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    """
    existing_data = {}
    if (
        reuse_existing_outputs
        and output_path is not None
        and os.path.exists(output_path)
    ):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i : i + batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = [{"role": "user", "content": instance["prompt"]}]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_chat_requesets(
                        messages_list=messages_list,
                        model=engine,
                        **completion_kwargs,
                    )
                )
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30*retry_count} seconds.")
                time.sleep(30 * retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(
                f"Failed to get response from OpenAI API after {retry_limit} retries."
            )
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output["choices"][0]["message"]["content"]
            instance["response_metadata"] = output
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results


# def get_prompt(
#     question, pattern_type, output_type, demos=None, stop="\n\n", add_question=False
# ):
#     def format_sample(question, output=None):
#         pass
#
#     template = INFERENCE_PATTERN[output_type]
#     prompt_input, prompt_input_output = (
#         PROMPT_DICT[pattern_type]["prompt_input"],
#         PROMPT_DICT[pattern_type]["prompt_input_output"],
#     )
#     if demos is None:
#         input = get_subtemp(template[0], {"question": question})
#
#     pass


def format_sample(
    sample,
    pattern_type,
    use_output=False,
    only_output=False,
    no_instruct=False,
    output_type="gsm8k",
):
    template = INFERENCE_PATTERN[output_type]
    append_output = not template[1].startswith("{")
    prompt_input, prompt_input_output = (
        PROMPT_DICT[pattern_type]["prompt_input"],
        PROMPT_DICT[pattern_type]["prompt_input_output"],
    )
    if not no_instruct:
        if not only_output:
            input = get_subtemp(template[0], {"question": sample["input"]})
        if use_output:
            output = get_subtemp(
                template[1],
                {
                    "chain_of_thought": sample["output"],
                    "answer": sample.get("answer", ""),
                },
            )
            if only_output:
                return output
    else:
        input = sample["input"]
        if use_output:
            output = sample["output"]

    if use_output:
        ret_sample = prompt_input_output.format(instruction=input, output=output)
    elif not append_output:
        ret_sample = prompt_input.format(instruction=input)
    else:
        ret_sample = prompt_input_output.format(
            instruction=input, output=template[1].split("{")[0]
        )
    return ret_sample


def delete_extra_zero(n):
    """删除小数点后多余的0"""
    try:
        n = float(n)
    except:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip("0")  # 删除小数点后多余的0
        n = int(n.rstrip(".")) if n.endswith(".") else float(n)  # 只剩小数点直接转int，否则转回float
        n = str(n)
        return n


def recover_options(input_str: str, combined: bool = False):
    options = input_str.split("Answer Choices:")[-1].strip()
    if "Let's" in options:
        options = options[: options.index("Let's")]

    if combined:
        return options
    else:
        index_1, index_2, index_3, index_4 = (
            options.find("(A)"),
            options.find("(B)"),
            options.find("(C)"),
            options.find("(D)"),
        )
        if "(E)" in options:
            index5 = options.find("(E)")

        opion_a = options[index_1 + 3 : index_2].strip()
        opion_b = options[index_2 + 3 : index_3].strip()
        opion_c = options[index_3 + 3 : index_4].strip()
        if "(E)" in options:
            opion_d = options[index_4 + 3 : index5].strip()
            option_e = [options[index5 + 3 :].strip()]
        else:
            opion_d = options[index_4 + 3 :].strip()
            option_e = []

        return [opion_a, opion_b, opion_c, opion_d] + option_e

def exact_match(
    prediction,
    reference,
    ignore_case=True,
    ignore_punctuation=True,
):
    predictions = np.array([prediction])
    references = np.array([reference])
    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    return predictions[0] == references[0]