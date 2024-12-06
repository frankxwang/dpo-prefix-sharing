from datasets import load_dataset
from sglang import function, system, user, assistant, assistant_begin, gen, set_default_backend, RuntimeEndpoint
from sglang.lang.chat_template import get_chat_template
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle as pkl
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

endpoint = RuntimeEndpoint(
    base_url="http://localhost:30000",
)
endpoint.chat_template = get_chat_template("llama-3-instruct")

set_default_backend(endpoint)

gsm8k = load_dataset("openai/gsm8k", "main")

gsm8k_cot_prompt = "Given the following problem, reason and give a final answer to the problem. Seperate each reasoning step with a single newline.\nProblem: {question}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n"

fork_amount = 2
max_outputs = 64
max_depth = 6

all_results = defaultdict(list)

@function
def answer_recursive(s, f, depth, question_id):
    if len(all_results[question_id]) >= max_outputs:
        return
    if depth == max_depth:
        f += gen("answer", max_tokens=128, temperature=0.6, top_p=0.9)
        finish_reason = f.get_meta_info('answer')['finish_reason']
        if 'matched' in finish_reason:
            f += "<|eot_id|>"
            all_results[question_id].append(f.text())
        return
    forks = f.fork(fork_amount)
    for fork in forks:
        fork += gen("answer", max_tokens=128, temperature=0.6, stop="\n", top_p=0.9)
        finish_reason = fork.get_meta_info('answer')['finish_reason']
        if 'matched' in finish_reason and finish_reason['matched'] == "\n":
            fork += "\n"
            answer_recursive.run(f=fork, depth=depth+1, question_id=question_id)
        elif 'matched' not in finish_reason: # usually this means max_tokens is hit or something
            print(finish_reason)
            print(fork["answer"])
            print(fork.get_meta_info('answer'))
            # exit()
        else:
            fork += "<|eot_id|>"
            all_results[question_id].append(fork.text())

@function
def answer_question(s, question, question_id):
    formatted_prompt = gsm8k_cot_prompt.format(question=question)
    s += user(formatted_prompt)
    s += assistant_begin()
    answer_recursive.run(f=s, depth=1, question_id=question_id)

def parse_outputs(output):
    pattern = r"(?s)(?<=\*\*START OUTPUT\*\*:).*?(?=\*\*END OUTPUT\*\*)"
    return [text.strip() for text in re.findall(pattern, output)]


batch_size = 128
for i in tqdm(range(0, len(gsm8k["train"]), batch_size)):
    all_results = defaultdict(list)
    state = answer_question.run_batch([dict(question=gsm8k["train"][question_id]["question"], question_id=question_id) for question_id in range(i, min(i+batch_size, len(gsm8k["train"])))])
    with open(f"round3-dpo-data/output_train_{i}.pkl", "wb") as f:
        pkl.dump(all_results, f)
    # print(all_results[::32])
    # print(state.text())
    # print(parse_outputs(state.text()))
