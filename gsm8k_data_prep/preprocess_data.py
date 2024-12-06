from datasets import load_dataset
import pickle as pkl
import re
import numpy as np
import pickle as pkl
import textwrap
from typing import List, Dict
from transformers import AutoTokenizer
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

class DictSequence(dict):
    def __getitem__(self, key):
        if isinstance(key, slice):
            return DictSequence((k, v[key]) for k, v in self.items())
        return super().__getitem__(key)
    def __len__(self):
        return len(self["tokens"])

def extract_single_number(s):
    s = s.replace(",", "")
    # Find all numbers in the string
    numbers = re.findall(r'\d+', s)

    # if len(numbers) != 1:
    if len(numbers) == 0:
        return None
    
    # Convert the single number to an integer and return it
    return int(numbers[-1])

def is_correct_gsm8k(sequence, answer):
    string = tokenizer.decode(sequence["tokens"])
    num = extract_single_number(string)
    return num is not None and (num == answer)

class RadixNode:
    def __init__(self, label=None):
        self.model_inputs = label  # Label on the node, contains sequence info for the model
        self.loss_inputs = dict() # set after the tree is constructed
        self.children = []
        self.is_end = False  # Is this node the end of a word
        self.reward = None

def insert(node, word):
    current = node
    while True:
        # Find the child whose label shares a common prefix with word
        for child in current.children:
            common_prefix_len = common_prefix_length(child.model_inputs, word)
            if common_prefix_len > 0:
                # There is a common prefix between the child's label and word
                if common_prefix_len == len(child.model_inputs):
                    # The child's label is fully matched
                    word = word[common_prefix_len:]
                    current = child
                    if len(word) == 0:
                        # The word is fully inserted
                        # for now, we will ignore the scenario where the model has one answer that is a total subset of another one
                        # current.is_end = True
                        return current
                    break  # Break out of for loop to continue with this child
                else:
                    # Need to split the child
                    # Create new child with the non-matching part of the child's label
                    new_child = RadixNode(child.model_inputs[common_prefix_len:])
                    new_child.children = child.children
                    new_child.is_end = child.is_end
                    new_child.reward = child.reward
                    # Update the child
                    child.model_inputs = child.model_inputs[:common_prefix_len]
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
                        # for now, we will ignore the scenario where the model has one answer that is a total subset of another one
                        # child.is_end = True
                        return child
        else:
            # No matching child found, add new child
            new_child = RadixNode(word)
            new_child.is_end = True
            current.children.append(new_child)
            current.is_end = False
            return new_child

def common_prefix_length(s1, s2):
    # Return length of common prefix between s1 and s2
    min_len = min(len(s1["tokens"]), len(s2["tokens"]))
    not_equals = s1["tokens"][:min_len] != s2["tokens"][:min_len]
    if np.sum(not_equals) == 0:
        return min_len
    return np.argmax(not_equals)

def make_tree(strings):
    tree = RadixNode()
    for val in strings:
        insert(tree, val)
    return tree

def print_tree(node, indent=''):
    if node.model_inputs is not None:
        scalar_keys = ["parent_index", "num_correct", "num_rollouts"]
        print(textwrap.fill(tokenizer.decode(node.model_inputs["tokens"]) + "[" + " ".join([f"{key}: {node.loss_inputs[key]}" for key in scalar_keys if key in node.loss_inputs]) + "]", initial_indent=indent, subsequent_indent=indent))
    for child in node.children:
        print_tree(child, indent + '  ')

def process_tree(root, gt_answer):
    preorder = []
    postorder = []
    parent_id = []
    good_outputs = []
    def process_tree_dfs(node):
        if not (node.is_end ^ (len(node.children) > 0)):
            breakpoint()
        preorder.append(node)
        if node.is_end:
            node.loss_inputs["num_correct"] = int(is_correct_gsm8k(node.model_inputs, gt_answer))
            node.loss_inputs["num_rollouts"] = 1
        else:
            node.loss_inputs["num_correct"] = 0
            node.loss_inputs["num_rollouts"] = 0
            parent_id.append(node.children)
            scores = []
            for child in node.children:
                process_tree_dfs(child)
                node.loss_inputs["num_correct"] += child.loss_inputs["num_correct"]
                node.loss_inputs["num_rollouts"] += child.loss_inputs["num_rollouts"]
                scores.append(child.loss_inputs["num_correct"] / child.loss_inputs["num_rollouts"])
            if not np.all(np.abs(np.array(scores) - scores[0]) < 0.01):
                good_outputs.append(node)
            # TODO: maybe do something with scores
        postorder.append(node)

    process_tree_dfs(root)
    if len(good_outputs) == 0:
        return None

    for i, node in enumerate(preorder):
        node.model_inputs["preorder_index"] = i
    for i, node in enumerate(postorder):
        node.model_inputs["postorder_index"] = i
    for i, nodes in enumerate(parent_id):
        for node in nodes:
            node.loss_inputs["parent_index"] = i
    root.loss_inputs["parent_index"] = -1

    model_inputs_keys = ["preorder_index", "postorder_index", "input_ids", "labels", "position_ids"]
    model_inputs = {key: [] for key in model_inputs_keys}
    model_inputs |= {"sequence_id": []}

    loss_inputs_keys = ["parent_index", "num_correct", "num_rollouts"]
    loss_inputs = {key: [] for key in loss_inputs_keys}
    loss_inputs |= {"parent_rollout_index": []}

    def append_node_to_data(node):
        for key in model_inputs_keys:
            value = node.model_inputs[key]
            if hasattr(value, "__iter__"):
                model_inputs[key].extend(value)
            else:
                model_inputs[key].extend([value] * len(node.model_inputs))

        node_id = len(loss_inputs["parent_index"])
        model_inputs["sequence_id"] += [node_id] * len(node.model_inputs)

        for key in loss_inputs_keys:
            value = node.loss_inputs[key]
            loss_inputs[key].append(value)

        parent_index = node.loss_inputs["parent_index"]
        num_prev_rollouts_from_the_parent = int(np.sum(np.array(loss_inputs["parent_index"]) == parent_index))
        loss_inputs["parent_rollout_index"].append(num_prev_rollouts_from_the_parent - 1)

    def flatten_tree(node):
        # we do this so that all children are grouped together
        for child in node.children:
            append_node_to_data(child)
        for child in node.children:
            flatten_tree(child)

    append_node_to_data(root)
    flatten_tree(root)

    return dict(
        model_inputs=model_inputs,
        loss_inputs=loss_inputs
    )

def preprocess_trajs(trajs, tokenizer):
    outputs = []
    for traj in trajs:
        all_tokens = np.array(tokenizer(traj)["input_ids"])
        input_ids = all_tokens[:-1]
        labels = all_tokens[1:]
        outputs.append(DictSequence(
            input_ids=input_ids,
            labels=labels,
            tokens=labels,
            position_ids=np.arange(len(input_ids)),
        ))
    return outputs


gsm8k = load_dataset("openai/gsm8k", "main")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

processed_data = []
for i in tqdm(range(0, len(gsm8k["train"]), 256)):
    with open(f"round2-dpo-data/output_train_{i}.pkl", "rb") as f:
        trajs_dict = pkl.load(f)
        for question_id, trajs in trajs_dict.items():
            if len(trajs) == 0:
                continue
            gt_answer = int(gsm8k["train"][question_id]["answer"].split("####")[-1].strip().replace(",", ""))
            tree = make_tree(preprocess_trajs(trajs, tokenizer))
            tree = tree.children[0] # the actual root is the first child
            outputs = process_tree(tree, gt_answer)
            if outputs is not None:
                processed_data.append(outputs)

model_inputs_keys = processed_data[0]["model_inputs"].keys()
loss_inputs_keys = processed_data[0]["loss_inputs"].keys()

model_inputs_struct = pa.struct([
    (key, pa.list_(pa.int32()))
    for key in model_inputs_keys
])
loss_inputs_struct = pa.struct([
    (key, pa.list_(pa.int32()))
    for key in loss_inputs_keys
])
schema = pa.schema([
    ("model_inputs", model_inputs_struct),
    ("loss_inputs", loss_inputs_struct),
])

# Function to extract lists for each key within a struct
def extract_struct_field(data, struct_key):
    return [record[struct_key] for record in data]

# Extract data for model_inputs and loss_inputs
model_inputs_data = extract_struct_field(processed_data, 'model_inputs')
loss_inputs_data = extract_struct_field(processed_data, 'loss_inputs')

# Create PyArrow StructArrays
model_inputs_array = pa.array(model_inputs_data, type=model_inputs_struct)
loss_inputs_array = pa.array(loss_inputs_data, type=loss_inputs_struct)

# Create a PyArrow Table
table = pa.Table.from_arrays(
    [model_inputs_array, loss_inputs_array],
    schema=schema
)

pq.write_table(table, "formatted_trees_gsm8k_round-3.parquet")
dataset = load_dataset("parquet", data_files="formatted_trees_gsm8k_round-3.parquet")
dataset.push_to_hub("fxwang/easy-math-trees_v2-round-3") 