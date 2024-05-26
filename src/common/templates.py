import random
import dataclasses


@dataclasses.dataclass
class FewShotPattern:
    """Patterns for few-shot tasks.

    The few-shot input are composed by a few examplers followed by final_suffix:
    {exampler no. 1} + {exampler no. 2} + {exampler no. 3}... + {final_suffix}

    Each exampler has the following format:
    {inputs_prefix} + {inputs} + {x_y_delimiter} + {targets_prefix} + {targets} +
    {example_separator}
    """

    inputs: str
    targets: str
    inputs_prefix: str = ""
    targets_prefix: str = ""
    x_y_delimiter: str = "\n\n"
    example_separator: str = "\n\n\n"
    final_suffix: str = ""
    input_pattern: str = "{{inputs}}{final_suffix}"
    in_template_mix: bool = True

    @property
    def few_shot_kwargs(self):
        return dict(
            inputs_prefix=self.inputs_prefix,
            targets_prefix=self.targets_prefix,
            x_y_delimiter=self.x_y_delimiter,
            example_separator=self.example_separator,
            final_suffix=self.final_suffix,
            input_pattern=self.input_pattern,
        )

    @property
    def combined_inputs(self):
        return self.inputs_prefix + self.inputs + self.x_y_delimiter

    @property
    def combined_targets(self):
        return self.targets_prefix + self.targets + self.example_separator

    @property
    def combined_inputs_w_target_prefix(self):
        return (
            self.inputs_prefix
            + self.inputs
            + self.x_y_delimiter
            + (self.targets_prefix)
        )

    @property
    def combined_targets_wo_target_prefix(self):
        return self.targets + self.example_separator


PATTERNS = {
    "cot": [
        (
            "{question} Let's think first. Chain of thought:",
            "{chain_of_thought}\nTherefore, the answer is {answer}.",
        ),
        (
            "{question} Think carefully first, then make a decision:",
            "{chain_of_thought} So, the answer is {answer}.",
        ),
        (
            "{question} Let's be accurate as possible.",
            "{chain_of_thought}\nThe answer: {answer}.",
        ),
        (
            "{question} Give me reasons, before answering the question",
            "{chain_of_thought} So the final answer is {answer}.",
        ),
        (
            "Lizzy: {question}.\nMe: Hmmm, let me think. I think this is the detailed solution:",
            "{chain_of_thought} Final answer: {answer}.",
        ),
        (
            "Question: {question} Think carefully first, then make a decision:",
            "{chain_of_thought} So the answer is {answer}.",
        ),
        (
            "Give the step-by-step reasoning process and then the final answer. {question}",
            "{chain_of_thought}\nThe final answer: {answer}.",
        ),
        (
            "{question}\nThoughts? Step-by-step reasoning:",
            "{chain_of_thought}\nThus, the answer is {answer}.",
        ),
        (
            "My question is: {question} Your thoughts:",
            "{chain_of_thought} The final answer: {answer}.",
        ),
        (
            "{question} Let's answer step by step:",
            "{chain_of_thought} The answer: {answer}.",
        ),
        # new
        (
            "Question: {question}\nLet's think step by step",
            "{chain_of_thought} The answer is {answer}",
        ),
        (
            "{question} Let's break it down step by step:",
            "{chain_of_thought}\nTherefore, the answer is {answer}.",
        ),
        (
            "{question} Think through the problem methodically:",
            "{chain_of_thought} So, the answer is {answer}.",
        ),
        (
            "{question} Let's analyze it carefully for a solution:",
            "{chain_of_thought}\nThe answer is {answer}.",
        ),
        (
            "{question} Provide a detailed explanation before the answer:",
            "{chain_of_thought} Hence, the final answer is {answer}.",
        ),
        (
            "I have a question for you: {question}\nHere's my detailed solution:",
            "{chain_of_thought} Final answer: {answer}.",
        ),
        (
            "Consider this question: {question} Think carefully before deciding:",
            "{chain_of_thought} Therefore, the answer is {answer}.",
        ),
        (
            "Give a step-by-step reasoning process and then the final answer for {question}",
            "{chain_of_thought}\nThe final answer is {answer}.",
        ),
        (
            "Question: {question} Reason through it step by step:",
            "{chain_of_thought}\nThus, the answer is {answer}.",
        ),
        (
            "Here's my question: {question} What are your thoughts?",
            "{chain_of_thought} The final answer is {answer}.",
        ),
        (
            "{question} Let's provide a step-by-step solution:",
            "{chain_of_thought} The answer is {answer}.",
        ),
    ],
    "cot_inverse": [
        # CoT + Answer --> Question
        (
            "Given the following reasoning and answer, what was the question? "
            "{chain_of_thought}\n The answer: {answer}",
            "The question {question}",
        ),
        # CoT + Answer --> Question
        (
            "For this chain-of-thought reasoning and answer, what was the "
            "question?\n{chain_of_thought}\n A: {answer}",
            "Q: {question}",
        ),
        # Question + Answer --> CoT
        (
            "Consider the question. {question}\n What is the step-by-step "
            "reasoning process to arrive at the answer: {answer}?",
            "{chain_of_thought}",
        ),
        # Question + Answer --> CoT
        (
            "Question. {question}\nAnswer. {answer}\nWhat step-by-step "
            "reasoning justifies that answer?",
            "Reasoning: {chain_of_thought}",
        ),
        # Question + Answer --> CoT
        (
            "Q: {question}\nA: {answer}\nExplain how we arrive at this answer: ",
            "Explanation: {chain_of_thought}",
        ),
        # CoT --> Question + Answer
        (
            "Given the rationale, provide a reasonable question and answer. "
            "Step-by-step reasoning process: {chain_of_thought}\n The question "
            "and answer:",
            "{question}\nThe answer is {answer}",
        ),
        # CoT --> Question + Answer
        (
            "{chain_of_thought}\nThis justifies what answer for what question? Q "
            "& A: ",
            "{question}\n{answer}",
        ),
        # CoT --> Question + Answer
        (
            "{chain_of_thought}is the reasoning for what question and answer pair?",
            "Q: {question}\nA: {answer}",
        ),
        # Answer --> Question + CoT
        (
            "Come up with a question and reasoning that would justify this "
            "answer: {answer}",
            "The question is: {question}\n"
            "Step-by-step reasoning process: {chain_of_thought}\n",
        ),
        # Answer --> Question + CoT
        (
            "Creatively image a question and justification for this answer: "
            "{answer}",
            "The question is: {question}\nStep-by-step reasoning "
            "process: {chain_of_thought}\n",
        ),
        # CoT + Answer --> Question
        (
            "What was the question for this implicit rationale, and corresponding"
            " answer?\n{chain_of_thought}\n The answer: {answer}",
            "The question: {question}",
        ),
        # Question + Answer --> CoT
        (
            "Consider the question. {question}\n If the answer is '{answer}'; "
            "explain the reasoning:",
            "{chain_of_thought}",
        ),
        # Question + Answer --> CoT
        (
            "Explain simply why {answer} is the correct answer to: {question}. "
            "Explanation:",
            "{chain_of_thought}",
        ),
        # CoT --> Question + Answer
        (
            "Given the stream of consciousness rationale, provide a reasonable "
            "question and answer. Rationale: {chain_of_thought}\n The question "
            "and answer:",
            "{question}\nThe answer is {answer}",
        ),
        # CoT --> Question + Answer
        (
            "Stream of consciousness rationale: {chain_of_thought}\nThe question "
            "and answer pair are described below.",
            "Q: {question}\nA: {answer}",
        ),
        # CoT --> Question + Answer
        (
            "Reconstruct a question, answer pair from this explanation: "
            "{chain_of_thought}\n",
            "Q:{question}\nA:{answer}",
        ),
        # Answer --> Question + CoT
        (
            "Come up with a question and stream of consciousness reasoning that "
            "would justify this answer: {answer}",
            "The question is: {question}\n"
            "Stream of consciousness: {chain_of_thought}\n",
        ),
        # Answer --> Question + CoT
        (
            "Imagine a question and stream-of-consciousness explanation for which"
            " this is the answer: {answer}",
            "Question: {question}\n" "Stream-of-consciousness: {chain_of_thought}",
        ),
    ],
    "code": [
        ("{question} Let's use python to solve the math problem.", "```\n{code}\n```"),
        ("Question: {question}", "# solution in Python:\n{code}"),
        (
            "Using Python, write a program that solves the math problem: {question}.",
            "```python\n{code}\n```",
        ),
        (
            "To solve the given math problem ({question}), implement a Python function that returns the result.",
            "```\n{code}\n```",
        ),
        (
            "Q: {question} Write a Python script to find the solution.",
            "```python\n{code}\n```",
        ),
        (
            "{question}\nWrite a Python program to determine the answer to the above question:",
            "```# Solution in Python\n{code}\n```",
        ),
        (
            "You have the following question:\n{question} Your task is to write a Python function to solve it:",
            "# Python code\n{code}",
        ),
        ("Q: {question}\nTo answer this question, write a Python program:", "{code}"),
        (
            "Given the following mathematical equation, write a Python script to find the solution:\n{question}",
            "# Python solution\n{code}",
        ),
        (
            "Take a look at this question: {question} Write a Python program to determine the answer:",
            "```python\n{code}\n```",
        ),
    ],
    "code_inverse": [
        # Code -> Question
        (
            "Given a code for solving a math problem, what was the question? {code}",
            "The question is {question}",
        ),
        (
            "Given a code snippet that sorts an array, what is the question that the code aims to solve?\n{code}",
            "The question being solved is: {question}",
        ),
        (
            "Code: {code}. What was the corresponding question?",
            "The question is: {question}",
        ),
        # Question + Answer -> Code
        (
            "For the code and the answer corresponding to a math question, what was the question?\n# Python code{code}\nThe answer is {answer}",
            "Question: {question]}",
        ),
        (
            "If you have a question and its corresponding answer, generate the Python code to solve it\nQuestion: {question}\nAnswer: {answer}",
            "Python code to solve the question: \n{code}",
        ),
        (
            "Question: {question}. Answer: {answer}. Generate the code.",
            "Python code: {code}",
        ),
        # Code -> Question + Answer
        (
            "Consider the code\n```\n{code}\n```\nThis justifies what answer for what question?",
            "Q: {question}\nA: {answer}",
        ),
        (
            "Analyze the following code and determine and the corresponding question and answer:\n```\n{code}\n```",
            "The question is: {question}\nThe answer is: {answer}",
        ),
        (
            "Code: {code}. What is the question? What is the answer?",
            "Question: {question}. Answer: {answer}",
        ),
        # Answer -> Question + Code
        (
            "Given an answer to a math problem, what is the question and the Python code that corresponds to it?\nAnswer: {answer}",
            "The question corresponding to the answer is: {question}\nPython code to solve the question: \n{code}",
        ),
        (
            "Provided with an answer to a problem:\nAnswer: {answer}\nWhat is the question associated with it? Generate the code.",
            "The question related to the answer is: {question}\nPython code to solve the question: \n{code}",
        ),
        (
            "Answer: {answer}. What is the corresponding question? Generate the code.",
            "Question: {question}. Python code: {code}",
        ),
        # Question + Code -> Answer
        (
            "Suppose you have a question and a Python code snippet that solves it, what is the corresponding answer?\nQuestion: {question}\nPython code: \n{code}",
            "The answer to the question is: {answer}",
        ),
        (
            "If you have a question and a code snippet that solves it:\nQuestion: {question}\nCode: {code}\nWhat is the answer?",
            "The answer to the question is: {answer}",
        ),
        (
            "Question: {question}. Code: {code}. What is the answer?",
            "The answer is: {answer}",
        ),
    ],
    "proof": [
        ("Theorem: {title}\n{text}", "Proof:\n```\n{target}\n```"),
        (
            "Prove {title}, here is the detaied content:\n{text}\nLet's prove step by step:",
            "{target}",
        ),
        (
            "Theorem: {title}\nStatement: {text}\nProve it.",
            "Proof:\n```\n{target}\n```",
        ),
        ("To prove: {title}\nStatement: {text}\nFollow the steps below:", "{target}"),
        ("Prove the following theorem:\n'{title}'\n{text}", "Proof:\n{target}"),
        (
            "Given the theorem '{title}' described as follows:\n{text}\nApply logical reasoning to prove:",
            "{target}",
        ),
        ("Prove the theorem '{title}'.\nStatement:\n{text}\nProof steps:", "{target}"),
        (
            "Can you prove the following theorem?\n\nTitle: {title}\n\nStatement:\n{text}",
            "Proof:\n{target}",
        ),
        (
            "Given the theorem '{title}', show that it holds true.\n\nStatement:\n{text}\n\nProof:",
            "{target}",
        ),
        (
            "Consider the theorem '{title}' with the following details:\n{text}\nProve the theorem step by step:",
            "{target}",
        ),
        (
            "Given: {text}\nProve the theorem titled '{title}'.",
            "Proof:\n```\n{target}\n```",
        ),
        (
            "Please demonstrate the validity of the theorem:\n'{title}'\nBased on the statement:\n{text}",
            "Proof:\n```\n{target}\n```",
        ),
        (
            "Consider the following statement:\n{text}\nProve the theorem '{title}'.",
            "Proof:\n```\n{target}\n```",
        ),
    ],
    "proof_w_ref": [
        (
            "Theorem: {title}\n{text}\n{refs}",
            "Proof:\n```\n{target}\n```",
            "Reference {title}:\n{text}",
        ),
        (
            "Prove {title}, here is the detaied content:\n{text}\nYou can refer to the following references:\n{refs}",
            "Let's prove step by step:\n{target}",
            "{title}:\n{text}",
        ),
        (
            "Prove the theorem '{title}' given the following information:\n{text}\nYou can refer to the references provided below:\n{refs}",
            "Proof:\n{target}",
            "{title} - {text}",
        ),
        (
            "Theorem: {title}\n{text}\n{refs}",
            "Proof:\n```\n{target}\n```",
            "Related theorems `{title}`: {text}",
        ),
        (
            "Prove {title}:\n{text}\nYou may find the following theorems helpful:\n{refs}",
            "Proof:\n{target}",
            "{title}: {text}",
        ),
        (
            "To prove '{title}', follow the instructions below:\n{text}\nUse the provided references for guidance:\n{refs}",
            "Proof:\n{target}",
            "Reference {title}:\n{text}",
        ),
        (
            "Can you prove the theorem '{title}'?\n{text}\nPlease provide a step-by-step proof supported by the following references:\n{refs}",
            "Proof:\n{target}",
            "#### Reference {title}\n{text}",
        ),
        (
            "Your task is to prove the theorem '{title}'.\n{text}\nConsult the following references for guidance:\n{refs}",
            "Proof:\n{target}",
            "Reference {title}:\n{text}",
        ),
    ],
}

encoding_templates_w_input = [
    # input encoding template, output encoding template, weight
    ("{instruction}\n\n{input}\n\n", "{output}", 0.2),
    ("{instruction}\n{input}\n\n", "{output}", 0.1),
    ("{instruction}\n{input}\n", "{output}", 0.1),
    ("{instruction}\n\nInput: {input}\n\nOutput:", "{output}", 0.05),
    ("{instruction}\nInput: {input}\nOutput:", "{output}", 0.05),
    ("{instruction}\n{input}\n\nResponse:", "{output}", 0.05),
    ("{instruction}\n\nAdditional Context:\n{input}\n\nAnswer:", "{output}", 0.05),
    ("Task: {instruction}\nInput: {input}\nOutput:", "{output}", 0.05),
    ("Task: {instruction}\n\n{input}\n\n", "{output}", 0.05),
    ("Task: {instruction}\n\n{input}\n\nAnswer:", "{output}", 0.05),
    (
        "You need to complete the following task:\n\n{instruction}\n\n{input}\n\nAnswer:",
        "{output}",
        0.05,
    ),
    (
        "{instruction}\n\nNow complete the following instance -\nInput: {input}\nOutput:",
        "{output}",
        0.05,
    ),
    ("Instruction:{instruction}\n\nInput: {input}\n\n", "{output}", 0.05),
    (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:",
        "{output}",
        0.1,
    ),  # alpaca template
]

encoding_templates_wo_input = [
    ("{instruction}\n\n", "{output}", 0.2),
    ("{instruction}\n", "{output}", 0.1),
    ("{instruction}", "\n{output}", 0.1),
    ("{instruction} Output:", "{output}", 0.05),
    ("{instruction}\nResponse:", "{output}", 0.05),
    ("{instruction}\n\nAnswer:", "{output}", 0.05),
    ("Task: {instruction}\n\n", "{output}", 0.05),
    ("Instruction: {instruction}\n", "{output}", 0.05),
    ("Instruction: {instruction}\nOutput:", "{output}", 0.05),
    ("You need to complete the following task:\n\n{instruction}\n\n", "{output}", 0.05),
    ("Can you help with this?\n\n{instruction}\n", "{output}", 0.05),
    ("Plase answer the following request: {instruction}\nAnswer:", "{output}", 0.05),
    (
        "Tell me how would you respond to the following request.\n{instruction}\n",
        "{output}",
        0.05,
    ),
    (
        "Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:",
        "{output}",
        0.1,
    ),  # alpaca template
]


def encode_instruction_example(
    instruction, input, output, random_template=True, eos_token=None
):
    if random_template:
        if input is not None and input.strip() != "":
            # randomly choose a template with input
            prompt_template, completion_template, _ = random.choices(
                encoding_templates_w_input,
                weights=[w for _, _, w in encoding_templates_w_input],
            )[0]
            prompt = prompt_template.format(
                instruction=instruction.strip(), input=input.strip()
            )
            completion = completion_template.format(output=output.strip())
        else:
            # randomly choose a template without input
            prompt_template, completion_template, _ = random.choices(
                encoding_templates_wo_input,
                weights=[w for _, _, w in encoding_templates_wo_input],
            )[0]
            prompt = prompt_template.format(instruction=instruction.strip())
            completion = completion_template.format(output=output.strip())
    else:
        if input is not None and input.strip() != "":
            prompt = instruction.strip() + "\n\n" + input.strip() + "\n\n"
            completion = output.strip()
        else:
            prompt = instruction.strip() + "\n\n"
            completion = output.strip()

    data = {
        "prompt": prompt,
        "completion": completion + eos_token if eos_token else completion,
    }
    return data


INFERENCE_PATTERN = {
    # "gsm8k": (
    #     "Question: {question}",
    #     "Let's think step by step\n{chain_of_thought} The answer is: {answer}",
    # ),
    "gsm8k": (
        "Question: {question}\nLet's think step by step",
        "{chain_of_thought} The answer is: {answer}",
    ),
    "gsm8k_question": (
        "Question: {question}\nSolve the question step by step",
        "{chain_of_thought} The answer is: {answer}",
    ),
    "gsm8k-survey": (),
    "gsm8k-pal": ("Question: {question}", "# solution in Python:\n{chain_of_thought}"),
    "agieval-zs": (
        'Question:\n{question}\nChoices:\n{options}\n\nAnswer:\nOutput the correct answer in this format: "Answer is choice ()"',
        "{answer}",
    ),
    "agieval-zs1": (
        "Question: {question}\n{options}",
        "The answer is",
    ),
    "cot": (
        "Question: {question}\nLet's think step by step",
        "{chain_of_thought} The answer is: {answer}",
    ),
    "mwp_infix": (
        "Question: {question}\nSolution equation:",
        "{chain_of_thought}",
    ),
    "mwp_prefix": (
        "Question: {question}\nPre-order traversal:",
        "{chain_of_thought}",
    ),
    "pal": (
        "Question: {question}\n# solution in Python:",
        "{chain_of_thought}",
    ),
    "pal_question": (
        "Question: {question}\n# write a solution() function to solve the problem in Python:",
        "{chain_of_thought}",
    ),
    "pal_plain": (
        "Question: {question}\n# non-semantic solution in Python:",
        "{chain_of_thought}",
    ),
    "lila": (
        "Question: {question}\n# solution in Python and print the answer:",
        "{chain_of_thought}",
    ),
    "cot-pal": (
        "Step-by-step solution: {cot}\nConvert the step-by-step solution into a solution in Python:",
        "{pal}",
    ),
    "pal-cot": (
        "# solution in Python\n{pal}\nConvert the solution in Python into a step-by-step solution:",
        "{cot}",
    ),
    "expr_reverse": (
        "Expression: {expression}\nDesign a math word problem based on the answer expression:",
        "{question}",
    ),
    "code_reverse": (
        "# solution in Python\n{expression}\nDesign a math word problem based on the Python solution:",
        "{question}",
    ),
    "mammoth": (
        "{question}",
        "{chain_of_thought} The answer is {answer}",
    ),
    "mammoth_pal": (
        "{question} Let's write a program.",
        "{chain_of_thought} The answer is {answer}",
    ),
    "abel": (
        "Question:\n{question}\nAnswer:\nLet's think step by step.\n",
        "{chain_of_thought} #### {answer}",
    ),
    "metamath": ("{question}", "{chain_of_thought} The answer is: {answer}"),
    "metamath_re2": (
        "{question}\nRead the question again: {question}",
        "{chain_of_thought} The answer is: {answer}",
    ),
    "refine_cot": (
        "## Question\n{question}\n\n## Natural language solution\n",
        "{chain_of_thought} The answer is: {answer}",
    ),
    "refine_pal": (
        "## Question\n{question}\n\n## Python solution\n",
        "{chain_of_thought}",
    ),
    "refine_cot_cot": (
        "## Question\n{question}\n\n## Natural language solution\n{generation}\n\n## Refined natural language solution\n",
        "{chain_of_thought} The answer is: {answer}",
    ),
    "refine_pal_pal": (
        "## Question\n{question}\n\n## Python solution\n{generation}\n\n## Refined Python solution\n",
        "{chain_of_thought}",
    ),
    "refine_cot_pal": (
        "## Question\n{question}\n\n## Natural language solution\n{generation}\n\n## Refined Python solution\n",
        "{chain_of_thought}",
    ),
    "refine_pal_cot": (
        "## Question\n{question}\n\n## Python solution\n{generation}\n\n## Refined natural language solution\n",
        "{chain_of_thought}",
    ),
    "strategy_cot": (
        "## Question\n{question}\n\n## Strategy\n{strategy}\n\n## Natural language solution\n",
        "{chain_of_thought}",
    ),
    "strategy_cot_loss": (
        "## Question\n{question}\n\n## Strategy\n",
        "{strategy}\n\n## Natural language solution\n{chain_of_thought}",
    ),
    "strategy_pal": (
        "## Question\n{question}\n\n## Strategy\n{strategy}\n\n## Python solution\n",
        "{chain_of_thought}",
    ),
    "strategy_pal_loss": (
        "## Question\n{question}\n\n## Strategy\n",
        "{strategy}\n\n## Python solution\n{chain_of_thought}",
    ),
    "feedback_v0": (
        "## Question\n{question}\n\n## Solution\n{generation}\n\n## Feedback\n",
        "{feedback}\n\n## Refined Solution\n{chain_of_thought}",
    ),
    "feedback_v00": (
        "## Question\n{question}\n\n## Solution\n{generation}\n\n## Feedback\n",
        "{feedback}\n\n## Refined natural langauge solution\n{chain_of_thought}",
    ),
    "feedback_code_v0": (
        "## Question\n{question}\n\n## Solution\n{generation}\n\n## Feedback\n",
        "{feedback}\n\n## Refined Python Solution\n{chain_of_thought}",
    ),
    "question_only": (
        "## Question\n{question}\n\n",
        "{chain_of_thought}",
    ),
    "nothing": (
        "{question}",
        "{chain_of_thought}",
    ),
    "question_only_sol": (
        "## Question\n{question}\n\n## Solution\n",
        "{chain_of_thought}",
    ),
    "question_only_pal": (
        "## Question\n{question} Let's write a program.\n\n",
        "{chain_of_thought}",
    ),
    "question_only_re2": (
        "## Question\n{question}\n\n## Parsed Question\n{question}\n\n",
        "{chain_of_thought}",
    ),
    "zs_cot": (
        "Question: {question}\nAnswer: Let's think step by step\n",
        "{chain_of_thought}",
    ),
    "zs_cot_question": (
        "Question: {question}",
        "Answer: Let's think step by step.\n{chain_of_thought}",
    ),
    "zs_cot_re2": (
        "Question: {question}\nRead the question again: {question}\nAnswer: Let's think step by step\n",
        "{chain_of_thought}",
    ),
    "zs_cot_ps": (
        "Question: {question}\nAnswer: Let's first understand the problem and devise a plan to solve the problem. "
        "Then, let's carry out the plan to solve the problem step by step.\n",
        "{chain_of_thought}",
    ),
    "faith_cot": (
        "## Question\n{question}\n\n## Strategy\n{strategy}\n\n## Python solution\n"
    ),
    "fs_pal": ("Question: {question}\n", "```python\n{chain_of_thought}\n```"),
    "math_corpus_cot": (
        "## Question\n{question}\n\n## Step-by-Step Solution\n",
        "{chain_of_thought}",
    ),
    "math_corpus_tora": (
        "## Question\n{question}\n\n## Code Solution\n",
        "{chain_of_thought}",
    ),
    "math_corpus_refine_cot": (
        "## Question\n{input}\n\n## Previous Step-by-Step Solution\n{old_output}\n\n## Feedback\n",
        "{feedback}\n\n## Refined Step-by-Step Solution\n{output}",
        "{feedback}",
    ),
    "math_corpus_refine_tora": (
        "## Question\n{input}\n\n## Previous Code Solution\n{old_output}\n\n## Feedback\n",
        "{feedback}\n\n## Refined Code Solution\n{output}",
        "{feedback}",
    ),
}

PROMPT_DICT = {
    "inst": {
        "prompt_input": ("{instruction}\n\n### Response:\n"),
        "prompt_input_output": ("{instruction}\n\n### Response:\n{output}"),
        "prompt_no_input": ("{instruction}\n\n### Response:\n"),
        "prompt_only_output": ("{output}"),
    },
    "no_inst": {
        "prompt_input": ("{instruction}\n\n"),
        "prompt_input_output": ("{instruction}\n\n{output}"),
        "prompt_no_input": ("{instruction}\n\n"),
        "prompt_only_output": ("{output}"),
    },
    "pure_no_inst": {
        "prompt_input": ("{instruction}"),
        "prompt_input_output": ("{instruction}{output}"),
        "prompt_no_input": ("{instruction}"),
        "prompt_only_output": ("{output}"),
    },
    "llama": {
        "prompt_input": ("{instruction}\n"),
        "prompt_input_output": ("{instruction}\n{output}"),
        "prompt_no_input": ("{instruction}\n\n### Response:"),
        "prompt_only_output": ("{output}"),
    },
    "alpaca": {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    "vicuna": {
        "prompt_input": (
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT: "
        ),
        "prompt_input_output": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:{output}"
        ),
    },
    "bayling": {
        "prompt_input": (
            "I am an intelligent language assistant developed by the NLP Group of ICT/CAS.\nBelow is a dialog consisting of instructions and responses. Write a response that completes the request.\n\n"
            "### Instruction:\n{instruction}\n### Response:\n"
        ),
        "prompt_input_output": (
            "I am an intelligent language assistant developed by the NLP Group of ICT/CAS.\nBelow is a dialog consisting of instructions and responses. Write a response that completes the request.\n\n"
            "### Instruction:\n{instruction}\n### Response:\n{output}"
        ),
    },
    "mammoth": {
        "prompt_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
        "prompt_input_output": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:{output}"
        ),
    },
    "mammoth_pot": {
        "prompt_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction} Let's write a program.\n\n### Response:"
        ),
        "prompt_input_output": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction} Let's write a program.\n\n### Response:{output}"
        ),
    },
    "abel": {
        "prompt_input": ("{instruction}"),
        "prompt_input_output": ("{instruction}{output}"),
    },
    "metamath": {
        "prompt_input": (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        ),
        "prompt_input_output": (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step.{output}"
        ),
    },
    "metamath_train": {
        "prompt_input": (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step.\n"
        ),
        "prompt_input_output": (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step.\n{output}"
        ),
    },
    "strategy": {
        "prompt_input": "## Question\n{input}\n\n## Strategy\n{instruction}\n\n## Solution:\n",
        "prompt_input_output": "## Question\n{input}\n\n## Strategy\n{instruction}\n\n## Solution:\n{output}",
        "prompt_no_input": "## Question\n{input}\n\n## Strategy\n{instruction}\n\n## Solution:\n",
        "prompt_only_output": "{output}",
    },
    "llama-chat": {
        "prompt_input": "[INST] <<SYS>>\nYou are an expert for math problem solving.\n<</SYS>>\n\n{instruction} [/INST]",
        "prompt_input_output": "[INST] <<SYS>>\nYou are an expert for math problem solving.\n<</SYS>>\n\n{instruction} [/INST] {output}",
    },
    "llama-qa": {
        "prompt_input": "[INST] Answer these questions, your answer should be as simple as possible, start your answer with the prompt 'The answer is '.\nQuestion: {instruction} [/INST] Answer:",
        "prompt_input_output": "[INST] {instruction} [/INST] Answer:{output}",
    },
    "llama-chat-out": {
        "prompt_input": "[INST] <<SYS>>\nYou are an expert for math problem solving.\n<</SYS>>\n\n{instruction} [/INST] Answer: Let's think step by step.\n",
        "prompt_input_output": "[INST] <<SYS>>\nYou are an expert for math problem solving.\n<</SYS>>\n\n{instruction} [/INST] {output}",
    },
    "llama-chat1": {
        "prompt_input": "[INST] <<SYS>>\nYou are an expert for math problem solving.\n<</SYS>>\n\n{instruction}\nAnswer: Let's think step by step.\n[/INST]",
        "prompt_input_output": "[INST] <<SYS>>\nYou are an expert for math problem solving.\n<</SYS>>\n\n{instruction} [/INST] {output}",
    },
    "llama-chat2": {
        "prompt_input": "[INST] <<SYS>>\nYou are an expert for math problem solving.\n<</SYS>>\n\nHelp me solve the problem.\n{instruction} [/INST]",
        "prompt_input_output": "[INST] <<SYS>>\nYou are an expert for math problem solving.\n<</SYS>>\n\n{instruction} [/INST] {output}",
    },
    "llama-chat3": {
        "prompt_input": "[INST] <<SYS>>\nYou are an expert for math problem solving.\n<</SYS>>\n\nSolve the following problem step by step.\n{instruction} [/INST]",
        "prompt_input_output": "[INST] <<SYS>>\nYou are an expert for math problem solving.\n<</SYS>>\n\n{instruction} [/INST] {output}",
    },
    "llama-chat-choice": {
        "prompt_input": "[INST] <<SYS>>\nChoose the correct option.\n<</SYS>>\n\n{instruction} [/INST]",
        "prompt_input_output": "[INST] <<SYS>>\nChoose the correct option.\n<</SYS>>\n\n{instruction} [/INST] {output}",
    },
    "llama-chat-humaneval": {
        "prompt_input": "[INST] <<SYS>>\nYou are an expert for coding\n<</SYS>>\n\nComplete the following python function.\n\n\n{instruction} [/INST] Here is the completed function:\n\n\n{instruction}",
        "prompt_input_output": "[INST] <<SYS>>\nYou are an expert for coding\n<</SYS>>\n\nComplete the following python function.\n\n\n{instruction} [/INST] Here is the completed function:\n\n\n{instruction}{output}",
    },
    "mistral-chat": {
        "prompt_input": "[INST] {instruction} [/INST]",
        "prompt_input_output": "[INST] {instruction} [/INST] {output}",
    },
    "mistral-chat-out": {
        "prompt_input": "[INST] {instruction} [/INST] Answer: Let's think step by step.\n",
        "prompt_input_output": "[INST] {instruction} [/INST] {output}",
    },
    "mistral-chat1": {
        "prompt_input": "[INST] {instruction}\nAnswer: Let's think step by step [/INST]",
        "prompt_input_output": "[INST] {instruction} [/INST] {output}",
    },
    "mistral-chat2": {
        "prompt_input": "[INST] Help me solve the problem.\n{instruction} [/INST]",
        "prompt_input_output": "[INST] {instruction} [/INST] {output}",
    },
    "mistral-chat3": {
        "prompt_input": "[INST] Solve the following problem step by step.\n{instruction} [/INST]",
        "prompt_input_output": "[INST] {instruction} [/INST] {output}",
    },
    "mistral-chat-humaneval": {
        "prompt_input": "[INST] Complete the following python function.\n\n\n{instruction} [/INST] Here is the completed function:\n\n\n{instruction}",
        "prompt_input_output": "[INST] Complete the following python function.\n\n\n{instruction} [/INST] Here is the completed function:\n\n\n{instruction}{output}",
    },
    "humaneval": {
        "prompt_input": "{instruction}",
        "prompt_input_output": "{instruction}{output}",
    },
    "llama-chat-ppl": {
        "prompt_input": "[INST] {instruction}",
        "prompt_input_output": "[INST] {instruction}{output} [/INST]",
    },
}

CHAT_PROMPT_DICT = {
    "llama-chat-out": [
        {
            "role": "system",
            "content": "You are an expert for math problem solving.",
        },
    ],
    "mistral-chat-out": [],
}

DATA_TYPE_DICT = {
    "cot": ["cot"],
    "pal": ["pal"],
    "cot_pal": ["cot", "pal"],
    "cot_pal_expr": ["cot", "pal", "mwp_infix", "mwp_prefix"],
}
IGNORE_INDEX = -100

EVOL_METHODS = [
    "Add new constraints and requirements to the original problem, adding approximately 10 additional words.",
    "If the original problem can be solved with only a few reasoning steps, please add more reasoning steps.",
    "Create a novel question that combines with other mathematical knowledge in middle school or university level.",
    "Create a brand-new question that belong to the same domain as the Given Question but be even more rare.",
]

if __name__ == "__main__":
    print(PROMPT_DICT["mammoth"]["prompt_input"])
