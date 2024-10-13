# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, List, Optional, Sequence, TypeVar

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer


DatasetType = TypeVar("DatasetType", Dataset, DatasetDict)


def is_conversational(example: Dict[str, Any]) -> bool:
    r"""
    Check if the example is in a conversational format.

    Args:
        example (`Dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the
            dataset type.

    Returns:
        `bool`: `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True
    >>> example = {"prompt": "The sky is"})
    >>> is_conversational(example)
    False
    ```
    """
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    # It must have one of the supported keys
    if example_keys:
        key = example_keys.pop()  # take the first supported key
        maybe_messages = example[key]
        # It must be a list of messages,
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # Each message must a list of dictionaries with keys "role" and "content"
            if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
                return True

    return False


def apply_chat_template(example: Dict[str, List[Dict[str, str]]], tokenizer: PreTrainedTokenizer) -> Dict[str, str]:
    r"""
    Apply a chat template to a conversational example.

    For more details, see [`maybe_apply_chat_template`].
    """
    # Check that the example has the correct keys
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys not in [
        {"messages"},  # language modeling
        {"prompt"},  # prompt-only
        {"prompt", "completion"},  # prompt-completion
        {"prompt", "chosen", "rejected"},  # preference
        {"chosen", "rejected"},  # preference with implicit prompt
        {"prompt", "completion", "label"},  # unpaired preference
    ]:
        raise KeyError(f"Invalid keys in the example: {example_keys}")

    # Apply the chat template to the whole conversation
    if "messages" in example:
        messages = tokenizer.apply_chat_template(example["messages"], tokenize=False)

    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)

    # Apply the chat template to the entire prompt + completion
    if "prompt" in example:  # explicit prompt and prompt-completion case
        if "chosen" in example:
            prompt_chosen = tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False)
            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:  # explicit prompt
            prompt_rejected = tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False)
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"], tokenize=False
            )
            completion = prompt_completion[len(prompt) :]
    else:  # implicit prompt case
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(example["rejected"], tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
    if "prompt" in example:
        error_message = (
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt alone. This can indicate that the chat template is not supported by TRL."
            "\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
        )
        if "chosen" in example and not prompt_chosen.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_chosen))
        if "rejected" in example and not prompt_rejected.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_rejected))
        if "completion" in example and not prompt_completion.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_completion))

    # Extract the completion by removing the prompt part from the prompt-completion string
    output = {}
    if "messages" in example:
        output["text"] = messages
    if "prompt" in example:
        output["prompt"] = prompt
    if "chosen" in example:
        output["chosen"] = chosen
    if "rejected" in example:
        output["rejected"] = rejected
    if "completion" in example:
        output["completion"] = completion
    if "label" in example:
        output["label"] = example["label"]

    return output


def maybe_apply_chat_template(
    example: Dict[str, List[Dict[str, str]]], tokenizer: PreTrainedTokenizer
) -> Dict[str, str]:
    r"""
    If the example is in a conversational format, apply a chat template to it.

    Args:
        example (`Dict[str, List[Dict[str, str]]`):
            Dictionary representing a single data entry of a conversational dataset. Each data entry can have different
            keys depending on the dataset type. The supported dataset types are:

                - Language modeling dataset: `"messages"`.
                - Prompt-only dataset: `"prompt"`.
                - Prompt-completion dataset: `"prompt"` and `"completion"`.
                - Preference dataset: `"prompt"`, `"chosen"`, and `"rejected"`.
                - Preference dataset with implicit prompt: `"chosen"` and `"rejected"`.
                - Unpaired preference dataset: `"prompt"`, `"completion"`, and `"label"`.

            For keys `"messages"`, `"prompt"`, `"chosen"`, `"rejected"`, and `"completion"`, the values are lists of
            messages, where each message is a dictionary with keys `"role"` and `"content"`.

        tokenizer (`PreTrainedTokenizer`):
            The tokenizer to apply the chat template with.

    Returns:
        `Dict[str, str]`: The formatted example with the chat template applied.

    Note:
        This function does not alter the keys, except for Language modeling dataset, where `"messages"` is replaced by
        `"text"`.

    Example:

    ```python
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    >>> example = {
    ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
    ...     "completion": [{"role": "assistant", "content": "It is blue."}]
    ... }
    >>> apply_chat_template(example, tokenizer)
    {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}
    ```
    """
    if is_conversational(example):
        return apply_chat_template(example, tokenizer)
    else:
        return example