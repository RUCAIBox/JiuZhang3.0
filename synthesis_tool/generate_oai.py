from datasets import load_from_disk
import random
import json


import json
import openai
from tqdm import tqdm


def get_response(prompt):
    messages=[{"role": "user", "content": prompt}]
    return openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
    )['choices'][0]["message"]["content"]


prompts = []

from datasets import load_from_disk
dataset = load_from_disk('/path/to/data')
for d in dataset:
    prompts.append(d['prompt'])

responses = []
max_retry = 3
with open("/path/to/save.jsonl", "w") as f:
    for prompt in tqdm(prompts):
        retry = 0
        while retry < max_retry:
            try:
                response = get_response(prompt)
                responses.append(response)
                f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
                break
            except:
                retry += 1
                continue



