"""Tools to generate from OpenAI prompts."""

import asyncio
import logging
import tiktoken
import os
from typing import Any

import aiolimiter
import openai
from aiohttp import ClientSession
from openai import error
from tqdm.asyncio import tqdm_asyncio
import random

with open("openai_api_key", "r") as f:
    openai_keys = f.readlines()
random.shuffle(openai_keys)
key_idx = 0
if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    openai.api_key = openai_keys[key_idx].strip()
    print(f"Use {openai.api_key}")

from zeno_build.models import lm_config
from zeno_build.prompts import chat_prompt

ERROR_ERRORS_TO_MESSAGES = {
    error.InvalidRequestError: "OpenAI API Invalid Request: Prompt was filtered",
    error.RateLimitError: "OpenAI API rate limit exceeded. Sleeping for 10 seconds.",
    error.APIConnectionError: "OpenAI API Connection Error: Error Communicating with OpenAI",  # noqa E501
    error.Timeout: "OpenAI APITimeout Error: OpenAI Timeout",
    error.ServiceUnavailableError: "OpenAI service unavailable error: {e}",
    error.APIError: "OpenAI API error: {e}",
}


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    n: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await openai.Completion.acreate(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    top_p=top_p,
                )
            except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:
                if isinstance(e, (error.ServiceUnavailableError, error.APIError)):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                elif isinstance(e, error.InvalidRequestError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Invalid Request: Prompt was filtered"
                                }
                            }
                        ]
                    }
                else:
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_completion(
    prompts: list[str],
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    n: int,
    top_p: float,
    requests_per_minute: int = 150,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: List of prompts to generate from.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of completions to generate for each API call.
        top_p: Top p to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=model_config.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    return [x["choices"][0]["text"] for x in responses]


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    limiter: aiolimiter.AsyncLimiter,
    temperature: float = 1.0,
    max_tokens: int = 512,
    n: int = 1,
    stop: list[str] = None,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    stop=stop,
                )
            except tuple(ERROR_ERRORS_TO_MESSAGES.keys()) as e:
                if isinstance(e, (error.ServiceUnavailableError, error.APIError)):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                elif isinstance(e, (error.RateLimitError)):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                    print("Retrying...", e)
                    key_idx += 1
                    if key_idx == len(openai_keys):
                        if "You exceeded your current quota" in str(e):
                            raise RuntimeError("All key is down")
                        key_idx = 0
                    openai.api_key = openai_keys[key_idx].strip()
                    print(f"Switch to {openai.api_key}")
                elif isinstance(e, error.InvalidRequestError):
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "Invalid Request: Prompt was filtered"
                                }
                            }
                        ]
                    }
                else:
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    full_messages: list[chat_prompt.ChatMessages],
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    n: int,
    top_p: float,
    requests_per_minute: int = 150,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        n: Number of responses to generate for each API call.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model_config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            top_p=top_p,
            limiter=limiter,
        )
        for messages in full_messages
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    return [x["choices"][0]["message"]["content"] for x in responses]


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    if isinstance(messages, dict):
        messages = [messages]
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4":
        print(
            "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "text-davinci-003":
        return len(encoding.encode(messages))
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
