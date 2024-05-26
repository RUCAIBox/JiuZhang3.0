import os
import openai
import asyncio
import time
import random
import aiolimiter
import openai
from aiohttp import ClientSession
from openai import error
from tqdm.asyncio import tqdm_asyncio
from src.evol.openai_utils import num_tokens_from_messages
from src.evol.openai_utils import _throttled_openai_chat_completion_acreate

# from vllm import SamplingParams

with open("openai_api_key", "r") as f:
    openai_keys = f.readlines()
random.shuffle(openai_keys)
key_idx = 0
if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    openai.api_key = openai_keys[key_idx].strip()
    print(f"Use {openai.api_key}")


def batch_call_gpt(
    prompts,
    bootstrap=None,
    model="code-davinci-002",
    stop=None,
    max_tokens=128,
    temperature=0.0,
    top_p=1.0,
    num_beams=1,
    replies=None,
    return_dict=True,
    logprobs=None,
):
    num_beams_batch_size = 20
    num_prompts = len(prompts)
    prompts = prompts * num_beams

    all_choices = []
    for batch_id in range(len(prompts) // num_beams_batch_size + 1):
        for i in range(20):
            try:
                start_idx = batch_id * num_beams_batch_size
                end_idx = min(start_idx + num_beams_batch_size, len(prompts))
                curr_prompts = prompts[start_idx:end_idx]
                choices = call_text_completion(
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    prompt=curr_prompts,
                    temperature=temperature,
                    top_p=top_p,
                    n=1,
                    best_of=1,
                    return_dict=return_dict,
                    logprobs=logprobs,
                    do_handle_right_space=False,
                )
                for choice in choices:
                    choice["index"] = start_idx + choice["index"]
                all_choices.extend(choices)
                break
            except openai.error.RateLimitError as e:
                print("Retrying...", e)
                time.sleep(max(i + 1, 3))
            except openai.InvalidRequestError as e:
                print("Retrying...", e)
                return ["超出长度，无解"]
            except Exception as e:
                print("Retrying...", e)
                time.sleep(max(i + 1, 3))
    assert len(all_choices) == num_beams * num_prompts, "Failed to call GPT API"
    batch_choices = [[] for _ in range(num_prompts)]
    for choice in all_choices:
        batch_choices[choice["index"] % num_prompts].append(choice)
    return batch_choices


def call_text_completion(
    model="code-davinci-002",
    max_tokens=128,
    stop=None,
    prompt=None,
    temperature=0.0,
    top_p=1.0,
    n=1,
    best_of=1,
    logprobs=None,
    return_dict=False,
    do_handle_right_space=True,
):
    def handle_right_space(prompt):
        if isinstance(prompt, str):
            right_space = False
            if prompt.endswith(" "):
                right_space = True
                prompt = prompt.rstrip(" ")
            return prompt, [right_space]
        right_space = []
        ret_prompt = []
        for p in prompt:
            if p.endswith(" "):
                right_space.append(True)
                ret_prompt.append(p.rstrip(" "))
            else:
                right_space.append(False)
                ret_prompt.append(p)
        return prompt, right_space

    if do_handle_right_space:
        prompt, right_space = handle_right_space(prompt)
    ans = openai.Completion.create(
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        n=n,
        best_of=best_of,
        logprobs=logprobs,
    )
    if do_handle_right_space:
        for c, rs in zip(ans["choices"], right_space):
            c["text"] = (
                c["text"]
                if not rs or c["text"] == " " or c["text"] == ""
                else c["text"].lstrip(" ")
            )
    return [c["text"] for c in ans["choices"]] if not return_dict else ans["choices"]


def call_chat_completion(
    model="code-davinci-002",
    max_tokens=128,
    stop=None,
    prompt=None,
    temperature=0.0,
    top_p=1.0,
    n=1,
    best_of=1,
    bootstrap=None,
    replies=None,
    return_dict=False,
):
    messages = []
    global key_idx
    if bootstrap is not None:
        messages.append(bootstrap)
    messages.append({"role": "user", "content": prompt})
    if replies is not None:
        messages.extend({"role": "assistant", "content": r} for r in replies)
    # print("======")
    # for m in messages:
    #     print(m)
    # print("======")
    # print(messages[-1]["content"])
    ans = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
        top_p=top_p,
        n=n,
    )
    return (
        [c["message"]["content"] for c in ans["choices"]]
        if not return_dict
        else ans["choices"]
    )


def call_gpt(
    prompt,
    bootstrap=None,
    model="code-davinci-002",
    stop=None,
    max_tokens=128,
    temperature=0.0,
    top_p=1.0,
    num_beams=1,
    replies=None,
    return_dict=False,
    logprobs=None,
):
    global key_idx
    num_beams_batch_size = 20
    completions = []
    if num_beams == 0:
        return []

    for i in range(20 * (num_beams // num_beams_batch_size + 1)):
        try:
            requestion_beams = min(num_beams_batch_size, num_beams - len(completions))
            if "davinci" in model:
                outputs = call_text_completion(
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    n=requestion_beams,
                    best_of=requestion_beams,
                    logprobs=logprobs,
                    return_dict=return_dict,
                )
            elif "turbo" in model or "gpt-4" in model:
                outputs = call_chat_completion(
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    n=requestion_beams,
                    best_of=requestion_beams,
                    bootstrap=bootstrap,
                    replies=replies,
                    return_dict=return_dict,
                )
            else:
                outputs = call_claude_completion(
                    prompt=prompt, model=model, stop=stop, max_tokens=max_tokens
                )
            completions.extend(outputs)
            if len(completions) >= num_beams:
                return completions[:num_beams]
        except openai.error.RateLimitError as e:
            print("Retrying...", e)
            key_idx += 1
            if key_idx == len(openai_keys):
                if "You exceeded your current quota" in str(e):
                    raise RuntimeError("All key is down")
                key_idx = 0
            openai.api_key = openai_keys[key_idx].strip()
            print(f"Switch to {openai.api_key}")
            time.sleep(max(i + 1, 3))
        except openai.InvalidRequestError as e:
            print("Retrying...", e)
            return ["超出长度，无解"]
        except Exception as e:
            print("Retrying...", e)
            time.sleep(max(i + 1, 3))
    raise RuntimeError("Failed to call GPT API")


def call_claude_completion(
    prompt,
    model="claude-instant-v1",
    stop=None,
    max_tokens=512,
):
    if anthropic.HUMAN_PROMPT not in prompt:
        claude_prompt = f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}"
    else:
        claude_prompt = prompt
    stops = [anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT]
    if stop is not None:
        if isinstance(stop, str):
            stops.append(stop)
        else:
            stops += stop
    completion = anthropic.completions.create(
        prompt=claude_prompt,
        stop_sequences=stops,
        model=model,
        max_tokens_to_sample=max_tokens,
        temperature=0,
    )
    print(completion)
    return [completion.completion.strip()]
    # print(response["completion"])
    # return [response["completion"].strip()]


def openai_to_claude_prompt(messages):
    role_map = {
        "system": "Human",
        "user": "Human",
        "assistant": "Assistant",
    }
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        transformed_role = role_map[role]
        prompt += f"\n\n{transformed_role.capitalize()}: {content}"
    prompt += "\n\nAssistant: "
    return prompt


def call_chatgpt(
    messages,
    model="gpt-3.5-turbo",
    stop=None,
    max_tokens=128,
    temperature=0.0,
    num_beams=1,
):
    global key_idx
    num_beams_batch_size = 20
    completions = []
    if num_beams == 0:
        return []

    for i in range(20 * (num_beams // num_beams_batch_size + 1)):
        try:
            requestion_beams = min(num_beams_batch_size, num_beams - len(completions))
            if "claude" in model:
                outputs = call_claude_completion(
                    model=model,
                    prompt=openai_to_claude_prompt(messages),
                    max_tokens=max_tokens,
                    stop=stop,
                )
            else:
                res = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    n=requestion_beams,
                )
                outputs = [c["message"]["content"] for c in res["choices"]]
            completions.extend(outputs)
            if len(completions) >= num_beams:
                return completions[:num_beams]
        except openai.error.RateLimitError as e:
            print("Retrying...", e)
            key_idx += 1
            if key_idx == len(openai_keys):
                if "You exceeded your current quota" in str(e):
                    raise RuntimeError("All key is down")
                key_idx = 0
            openai.api_key = openai_keys[key_idx].strip()
            print(f"Switch to {openai.api_key}")
            time.sleep(max(i + 1, 3))
        except openai.InvalidRequestError as e:
            print("Retrying...", e)
            return ["超出长度，无解"]
        except Exception as e:
            print("Retrying...", e)
            if "该令牌额度已用尽" in str(e):
                key_idx += 1
                if key_idx == len(openai_keys):
                    if "You exceeded your current quota" in str(e):
                        raise RuntimeError("All key is down")
                    key_idx = 0
                openai.api_key = openai_keys[key_idx].strip()
                print(f"Switch to {openai.api_key}")
            time.sleep(max(i + 1, 3))
    raise RuntimeError("Failed to call GPT API")


async def dispatch_openai_requests(messages_list, model: str, temperature: float):
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(model=model, messages=x, temperature=temperature)
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def call_vllm(
    llm,
    prompt,
    stop=None,
    max_tokens=128,
    temperature=0.0,
    num_beams=1,
):
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, stop=stop, n=num_beams
    )
    outputs = llm.generate(prompt, sampling_params)
    return outputs[0].outputs.text


class LLM:
    def __init__(self) -> None:
        with open("openai_api_key", "r") as f:
            self.openai_keys = f.readlines()
        random.shuffle(openai_keys)
        self.key_idx = 0

    def switch_key(self):
        self.key_idx += 1
        if self.key_idx == len(self.openai_keys):
            raise RuntimeError("All key is down")
        openai.api_key = self.openai_keys[self.key_idx].strip()
        print(f"Switch to {openai.api_key}")

    def chat(
        self,
        messages,
        model="gpt-3.5-turbo",
        stop=None,
        max_tokens=128,
        temperature=0.0,
        num_beams=1,
    ):
        openai.api_key = openai_keys[key_idx].strip()
        num_beams_batch_size = 20
        completions = []
        if num_beams == 0:
            return []

        for i in range(20 * (num_beams // num_beams_batch_size + 1)):
            try:
                requestion_beams = min(
                    num_beams_batch_size, num_beams - len(completions)
                )
                if "claude" in model:
                    outputs = call_claude_completion(
                        model=model,
                        prompt=openai_to_claude_prompt(messages),
                        max_tokens=max_tokens,
                        stop=stop,
                    )
                else:
                    res = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        stop=stop,
                        temperature=temperature,
                        n=requestion_beams,
                    )
                    outputs = [c["message"]["content"] for c in res["choices"]]
                completions.extend(outputs)
                if len(completions) >= num_beams:
                    return completions[:num_beams]
            except openai.error.RateLimitError as e:
                print("Retrying...", e)
                time.sleep(max(i + 1, 3))
            except openai.InvalidRequestError as e:
                print("Retrying...", e)
                return ["超出长度，无解"]
            except Exception as e:
                print("Retrying...", e)
                if "该令牌额度已用尽" in str(e) or "The token quota has been used up" in str(e):
                    self.switch_key()
                time.sleep(max(i + 1, 3))
        raise RuntimeError("Failed to call GPT API")

    async def _throttled_openai_chat_completion_acreate(
        self,
        model: str,
        messages: list[dict[str, str]],
        limiter: aiolimiter.AsyncLimiter,
        temperature: float = 1.0,
        max_tokens: int = 512,
        n: int = 1,
        stop: list[str] = None,
    ) -> dict[str]:
        async with limiter:
            for _ in range(5):
                try:
                    return await openai.ChatCompletion.acreate(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=n,
                        stop=stop,
                    )
                except openai.error.RateLimitError as e:
                    print("Retrying...", e)
                    await asyncio.sleep(3)
                except openai.InvalidRequestError as e:
                    print("Retrying...", e)
                    return ["Invalid Request: Prompt was filtered"]
                except Exception as e:
                    print("Retrying...", e)
                    if "该令牌额度已用尽" in str(
                        e
                    ) or "The token quota has been used up" in str(e):
                        self.switch_key()
                    time.sleep(3)
            return {"choices": [{"message": {"content": ""}} for _ in range(n)]}

    async def achat(
        self,
        batch_messages,
        model="gpt-3.5-turbo",
        stop=None,
        max_tokens=128,
        temperature=0.0,
        num_beams=1,
        requests_per_minute: int = 150,
    ):
        openai.aiosession.set(ClientSession())
        limiter = aiolimiter.AsyncLimiter(requests_per_minute)
        async_responses = [
            self._throttled_openai_chat_completion_acreate(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=num_beams,
                stop=stop,
                limiter=limiter,
            )
            for messages in batch_messages
        ]
        responses = await tqdm_asyncio.gather(*async_responses)
        # Note: will never be none because it's set, but mypy doesn't know that.
        await openai.aiosession.get().close()  # type: ignore
        try:
            return [[c["message"]["content"] for c in x["choices"]] for x in responses]
        except:
            ret = []
            for x in responses:
                try:
                    ret.append([c["message"]["content"] for c in x["choices"]])
                except:
                    ret.append([""] * num_beams)
            return ret


if __name__ == "__main__":
    ans = batch_call_gpt(
        ["Once upon a time,"] * 10,
        bootstrap=None,
        model="curie",
        max_tokens=20,
        num_beams=5,
        return_dict=True,
    )
    print(ans)
