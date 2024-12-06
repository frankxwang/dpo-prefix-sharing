from datasets import load_dataset
from sglang import function, system, user, assistant, assistant_begin, gen, set_default_backend, RuntimeEndpoint
from sglang.lang.chat_template import get_chat_template
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle as pkl
from collections import defaultdict
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

endpoint = RuntimeEndpoint(
    base_url="http://localhost:30000",
)
endpoint.chat_template = get_chat_template("llama-3-instruct")

set_default_backend(endpoint)

gsm8k = load_dataset("openai/gsm8k", "main")

gsm8k_cot_prompt = "Given the following problem, reason and give a final answer to the problem. Seperate each reasoning step with a single newline.\nProblem: {question}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n"

all_results = defaultdict(list)

@function
def answer_question(s, question, question_id):
    formatted_prompt = gsm8k_cot_prompt.format(question=question)
    s += user(formatted_prompt)
    s += assistant_begin()
    s += gen("answer", max_tokens=1024, temperature=0)
    all_results[question_id].append(s.text())

def parse_outputs(output):
    pattern = r"(?s)(?<=\*\*START OUTPUT\*\*:).*?(?=\*\*END OUTPUT\*\*)"
    return [text.strip() for text in re.findall(pattern, output)]

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

correct = []
batch_size = 32
for i in tqdm(range(0, len(gsm8k["test"]), batch_size)):
    all_results = defaultdict(list)
    state = answer_question.run_batch([dict(question=gsm8k["test"][question_id]["question"], question_id=question_id) for question_id in range(i, min(i+batch_size, len(gsm8k["test"])))])
    with open(f"gsm8k_packing-llama-3p2-1B-round2/output_test_{i}.pkl", "wb") as f:
        pkl.dump(all_results, f)
    for question_id, responses in all_results.items():
        answer = gsm8k["test"][question_id]["answer"].split("####")[-1].strip()
        answer = answer.replace(",", "")
        for response in responses:
            correct.append(is_correct_gsm8k(response, int(answer)))
print(np.mean(correct))
    # print(state.text())
    # print(parse_outputs(state.text()))
