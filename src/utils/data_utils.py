import re
import random
import datasets
import pandas as pd


def unwrap_boxed(string):
    pattern = r"\\boxed{([^}]*)}"  # Regular expression pattern to match \boxed{}
    unwrapped_string = re.sub(pattern, r"\1", string)
    return unwrapped_string


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return ""
    return pred[-1]


def extract_answer_math(s):
    ans = s.split("boxed")
    if len(ans) == 1:
        return s
    ans = ans[-1]
    if len(ans) == 0:
        return ""
    try:
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
    except:
        return ""
    return a


def extract_choice_answer(text, answer_prefix="The answer is:"):
    answer = extract_cot_answer(text, answer_prefix=answer_prefix)
    tmp = re.findall(r"\b(A|B|C|D|E)\b", answer.upper())
    if tmp:
        pred = tmp
    else:
        pred = [answer.strip().strip(".")]
    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = answer
    else:
        # choose the last e
        pred = pred[-1]
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    return pred


def extract_flan_cot_answer(answer: str, is_target: bool = False, **unused_kwargs):
    """Postprocess target and answer containing china-of-thought process.

    Args:
      answer: text string from model output.
      example: input example.
      is_target: whether apply postprocess function on target.

    Returns:
      output: answer removing chain-of-thought texts.
    """
    answer = answer.strip()
    # corner case 1: unified_qa_science target always has a period at end.
    if answer[-1] in [".", ",", "?", " ", "\n"]:
        answer = answer[:-1].strip()

    if is_target:
        # corner case 2: target = (B), prediction = B.
        if answer[0] == "(" and answer[-1] == ")":
            answer = answer[1:-1].strip()
        return answer
    else:
        answer = answer.split("answer is")[-1].strip()
        answer = answer.split("final answer")[-1].strip()
        answer = answer.split("Final answer")[-1].strip()
        answer = answer.split("answer:")[-1].strip()
        answer = answer.split("Answer:")[-1].strip()
        if answer and answer[0] in [".", ",", "?", " ", "\n", ":"]:
            answer = answer[1:].strip()
        if answer and answer[-1] in [".", ",", "?", " ", "\n", ":"]:
            answer = answer[:-1].strip()
        # corner case 2: target = (B), prediction = B.
        if answer and answer[0] == "(" and answer[-1] == ")":
            answer = answer[1:-1].strip()
        # TODO(yunxuanli) corner case 3: target = (B), prediction = yes (option B)
        return answer


def extract_cot_answer(text, answer_prefix="The answer is: "):
    if isinstance(text, list):
        text = "\n".join([t["content"] for t in text])
    split_list = text.split(answer_prefix)
    if len(split_list) == 1:
        answer = extract_answer_math(text)
        if answer == "":
            answer = extract_answer_number(text)
    else:
        answer = split_list[-1].strip().strip("。.：:")
    return answer


def convert_to_number(string):
    try:
        number = int(string)
        return number
    except ValueError:
        try:
            number = float(string)
            return number
        except ValueError:
            return None


def get_subtemp(template, kvs):
    instance = template
    for key in kvs:
        instance = instance.replace("{" + key + "}", kvs[key])
    return instance


def get_instance(templates, kvs):
    template = random.choice(templates)
    input_ins = get_subtemp(template[0], kvs)
    output_ins = get_subtemp(template[1], kvs)
    return input_ins, output_ins


def get_instance_refs(templates, kvs):
    template = random.choice(templates)
    refs = "\n".join([get_subtemp(template[-1], ref) for ref in kvs["refs"]])
    kvs["refs"] = refs
    input_ins = get_subtemp(template[0], kvs)
    output_ins = get_subtemp(template[1], kvs)
    return input_ins, output_ins


def save_samples_to_dataset(samples, save_path):
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=samples))
    dataset.save_to_disk(save_path)


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r"\.{2,}"


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub("[.]" + digits, "<prd>\\1", text)
    text = re.sub(
        multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences
