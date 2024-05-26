# make contra pair, pos sample and neg sample
# create data/pairs/xxx.json

import os
import json
import tqdm
import random

from copy import deepcopy
import re
import nltk

# nltk.download("punkt")  # Download the necessary dataset for tokenization


# An expression tree node
class Et:
    # Constructor to create a node
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


# Returns root of constructed tree for given postfix expression
def construct_exp_tree(postfix):
    stack = []

    # Traverse through every character of input expression
    for char in postfix:
        # if operand, simply push into stack
        if char not in ["+", "-", "*", "/", "^"]:
            t = Et(char)
            stack.append(t)
        # Operator
        else:
            # Pop two top nodes
            t = Et(char)
            t1 = stack.pop()
            t2 = stack.pop()

            # make them children
            t.right = t1
            t.left = t2

            # Add this subexpression to stack
            stack.append(t)
    # Only element  will be the root of expression tree
    t = stack.pop()
    return t


def from_infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while (
                len(st) > 0
                and st[-1] not in ["(", "["]
                and priority[e] <= priority[st[-1]]
            ):
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res


def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while (
                len(st) > 0
                and st[-1] not in [")", "]"]
                and priority[e] < priority[st[-1]]
            ):
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res


def out_expression_list(test, output_lang, num_list, num_stack=None):
    max_index = output_lang.n_words
    res = []
    for i in test:
        # if i == 0:
        #     return res
        if i < max_index - 1:
            idx = output_lang.index2word[i]
            if idx[0] == "N":
                if int(idx[1:]) >= len(num_list):
                    return None
                res.append(num_list[int(idx[1:])])
            else:
                res.append(idx)
        else:
            pos_list = num_stack.pop()
            c = 1  # 兜底逻辑
            if len(num_list) > 0:
                c = num_list[0]
            if len(pos_list) > 0:
                c = num_list[pos_list[0]]
            res.append(c)
    return res


def compute_postfix_expression(post_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    for p in post_fix:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(
                    eval(p[pos.start() : pos.end() - 1] + "+" + p[pos.end() - 1 :])
                )
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if a == 0:
                return None
            st.append(b / a)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(b - a)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a**b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(
                    eval(p[pos.start() : pos.end() - 1] + "+" + p[pos.end() - 1 :])
                )
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a**b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def score(node):
    if not isinstance(node, list):
        return 1
    return score(node[1][0]) + score(node[1][1]) + 1


def tree_normalize(node):
    # pdb.set_trace()
    if not isinstance(node, list):
        return node
    score0 = score(node[1][0])
    score1 = score(node[1][1])

    if score0 > score1:
        tmp = node[1][1]
        node[1][1] = node[1][0]
        node[1][0] = tmp

    tree_normalize(node[1][0])
    tree_normalize(node[1][1])

    return node


def maketree(forword):
    # construct tree
    valid_op = ["+", "-", "*", "/"]
    root = None
    now = root
    for word in forword:
        if word in valid_op:
            node = [word, [None, None], None]
            if now == None:
                root = node
                now = root
            else:
                if now[1][0] == None:
                    now[1][0] = node
                    node[2] = now
                    now = node
                else:
                    now[1][1] = node
                    node[2] = now
                    now = node
        else:
            if now == None and root == None:
                root = word
            elif now != None:
                if now[1][0] == None:
                    now[1][0] = word
                else:
                    now[1][1] = word
                    while now != None and now[1][1] != None:
                        now = now[2]
    if now != None:
        print("error", forword)
    # root = tree_normalize(root)
    return root


def add_order(tree, order):
    if isinstance(tree, list):
        tree.append(order)
        new_order = add_order(tree[1][0], order + 1)
        new_order = add_order(tree[1][1], new_order)
        return new_order
    else:
        return order + 1


def comp(t1, t2):
    if not isinstance(t1, list) and not isinstance(t2, list):
        return True
    if isinstance(t1, list) and isinstance(t2, list) and t1[0] == t2[0]:
        return comp(t1[1][0], t2[1][0]) and comp(t1[1][1], t2[1][1])
    return False


def is_sub_tree(t, s):
    if not isinstance(t, list):
        return -1
    if not isinstance(s, list):
        return -1
    if comp(t, s):
        return s[3]
    else:
        x1 = is_sub_tree(t, s[1][0])
        if x1 != -1:
            return x1
        x2 = is_sub_tree(t, s[1][1])
        return x2


def common_tree(t, s):
    if not isinstance(t, list):
        return (-1, -1)
    if score(t) <= 3:
        return (-1, -1)
    x = is_sub_tree(t, s)
    if x != -1:
        return (t[3], x)

    x = common_tree(t[1][0], s)
    if x != (-1, -1):
        return x
    x = common_tree(t[1][1], s)
    return x


def func1(neg_idx, idx, ops, n_num):
    if ops[idx] == ops[neg_idx] and n_num[idx] != n_num[neg_idx]:
        return True
    return False


def func2(neg_idx, idx, ops, n_num):
    if ops[idx] != ops[neg_idx] and n_num[idx] == n_num[neg_idx]:
        return True
    return False


def RetTrue(*args):
    return True


def transfer_num(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["original_text"].strip().split(" ")
        if "segmented_text" in d:
            seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start() : pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end() :])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    # if nums.count(n) == 1:
                    #     res.append("N" + str(nums.index(n)))
                    # else:
                    #     res.append(n)
                    res.append(n)  # FIXME
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                # if nums.count(st_num) == 1:
                #     res.append("N" + str(nums.index(st_num)))
                # else:
                #     res.append(st_num)
                res.append(st_num)  # FIXME
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        #         generate_nums_dict[s] = 0
        #     if s in generate_nums and s not in nums:
        #         generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        # for i, j in enumerate(input_seq):
        #     if j == "NUM":
        #         num_pos.append(i)
        # assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 15:
            temp_g.append(g)
    print(temp_g)
    return pairs, temp_g, copy_nums


def transfer_num_general(data):  # transfer num into "NUM"
    print("Transfer numbers...")
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pairs = []
    generate_nums = []
    generate_nums_dict = {}
    copy_nums = 0
    for d in data:
        nums = []
        input_seq = []
        seg = d["original_text"].strip().split(" ")
        if "segmented_text" in d:
            seg = d["segmented_text"].strip().split(" ")
        equations = d["equation"][2:]

        for s in seg:
            pos = re.search(pattern, s)
            if pos and pos.start() == 0:
                nums.append(s[pos.start() : pos.end()])
                input_seq.append("NUM")
                if pos.end() < len(s):
                    input_seq.append(s[pos.end() :])
            else:
                input_seq.append(s)
        if copy_nums < len(nums):
            copy_nums = len(nums)

        nums_fraction = []

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    # if nums.count(n) == 1:
                    #     res.append("N" + str(nums.index(n)))
                    # else:
                    #     res.append(n)
                    res.append(n)  # FIXME
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                # if nums.count(st_num) == 1:
                #     res.append("N" + str(nums.index(st_num)))
                # else:
                #     res.append(st_num)
                res.append(st_num)  # FIXME
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        out_seq = seg_and_tag(equations)
        # for s in out_seq:  # tag the num which is generated
        #     if s[0].isdigit() and s not in generate_nums and s not in nums:
        #         generate_nums.append(s)
        #         generate_nums_dict[s] = 0
        #     if s in generate_nums and s not in nums:
        #         generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_pos = []
        # for i, j in enumerate(input_seq):
        #     if j == "NUM":
        #         num_pos.append(i)
        # assert len(nums) == len(num_pos)
        # pairs.append((input_seq, out_seq, nums, num_pos, d["ans"]))
        pairs.append((input_seq, out_seq, nums, num_pos))

    temp_g = []
    for g in generate_nums:
        if generate_nums_dict[g] >= 15:
            temp_g.append(g)
    print(temp_g)
    return pairs, temp_g, copy_nums

def remove_brackets(x):
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y

def extract_infix_answer(text):
    try:
        return str(eval(text.strip()))
    except:
        return ""


def extract_prefix_answer(text):
    try:
        return str(compute_prefix_expression(text.split()))
    except:
        return ""


if __name__ == "__main__":
    process_mathqa()
    # print(extract_prefix_answer(") /  * 100 / * 36 100 * 3 10 * 3 10"))
