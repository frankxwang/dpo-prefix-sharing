from datasets import load_dataset
import pickle as pkl
import re
import numpy as np
import pickle as pkl
import textwrap

def extract_single_number(s):
    s = s.replace(",", "")
    # Find all numbers in the string
    numbers = re.findall(r'\d+', s)

    # if len(numbers) != 1:
    if len(numbers) == 0:
        return None
    
    # Convert the single number to an integer and return it
    return int(numbers[-1])

def is_correct_gsm8k(string, answer):
    num = extract_single_number(string)
    return num is not None and (num == answer)

class RadixNode:
    def __init__(self, label=''):
        self.label = label  # Label on the edge to this node
        self.children = []
        self.is_end = False  # Is this node the end of a word
        self.reward = None

def insert(node, word):
    current = node
    while True:
        # Find the child whose label shares a common prefix with word
        for child in current.children:
            common_prefix_len = common_prefix_length(child.label, word)
            if common_prefix_len > 0:
                # There is a common prefix between the child's label and word
                if common_prefix_len == len(child.label):
                    # The child's label is fully matched
                    word = word[common_prefix_len:]
                    current = child
                    if len(word) == 0:
                        # The word is fully inserted
                        current.is_end = True
                        return current
                    break  # Break out of for loop to continue with this child
                else:
                    # Need to split the child
                    # Create new child with the non-matching part of the child's label
                    new_child = RadixNode(child.label[common_prefix_len:])
                    new_child.children = child.children
                    new_child.is_end = child.is_end
                    new_child.reward = child.reward
                    # Update the child
                    child.label = child.label[:common_prefix_len]
                    child.children = [new_child]
                    child.is_end = False
                    child.reward = None
                    # Now, if there is remaining part of word, add it as child
                    if len(word[common_prefix_len:]) > 0:
                        new_leaf = RadixNode(word[common_prefix_len:])
                        new_leaf.is_end = True
                        child.children.append(new_leaf)
                        return new_leaf
                    else:
                        child.is_end = True
                        return child
        else:
            # No matching child found, add new child
            new_child = RadixNode(word)
            new_child.is_end = True
            current.children.append(new_child)
            return new_child

def common_prefix_length(s1, s2, sep="\n"):
    # Return length of common prefix between s1 and s2
    s1 = s1.split(sep)
    s2 = s2.split(sep)
    min_len = min(len(s1), len(s2))
    total_prefix_len = 0
    for i in range(min_len):
        if s1[i] == s2[i]:
            total_prefix_len += len(s1[i]) + len(sep)
        else:
            break
    return total_prefix_len

def make_tree(strings, answer):
    tree = RadixNode()
    for val in strings:
        child = insert(tree, val)
        child.reward = is_correct_gsm8k(child.label, answer)
    return tree

def print_tree(node, indent=''):
    if node.label != '':
        print(textwrap.fill(node.label + (f' [reward: {node.reward}]' if node.is_end else ''), initial_indent=indent, subsequent_indent=indent))
    for child in node.children:
        print_tree(child, indent + '  ')

gsm8k = load_dataset("openai/gsm8k", "main")

results = []
for i in range(0, 1000, 256):
    with open(f"round2-dpo-data/output_train_{i}.pkl", "rb") as f:
        trajs_dict = pkl.load(f)
        for question_id, trajs in trajs_dict.items():
            answers = []
            gt_answer = int(gsm8k["train"][question_id]["answer"].split("####")[-1].strip().replace(",", ""))
            for traj in trajs:
                traj = traj.strip()
                model_answer = extract_single_number(traj.split("\n")[-1].replace(",", ""))
                answers.append(model_answer)
            # print(np.array(answers) == gt_answer)
            # results.append(np.all(np.array(answers) == gt_answer))
            # print(answers)
            # print_tree(make_tree(trajs, gt_answer))
            # print("GT_ANSWER:", gt_answer)
            # breakpoint()
            # if i == 1:
            #     exit()
            results.append(np.any(np.array(answers) == gt_answer) and np.any(np.array(answers) != gt_answer))
            # results.extend(np.array(answers) == gt_answer)
            # print(gt_answer, answers)

print(np.mean(results))