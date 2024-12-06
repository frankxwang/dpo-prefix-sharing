from typing import Optional, List, Dict, Any, Union, Literal
from dataclasses import dataclass
import torch
import numpy as np 
from trl.trainer.utils import pad


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1, padding_side: Literal["left", "right"] = "right") -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        ) if padding_side == "right" else torch.cat(
            [
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                tensor,
            ],
            dim=dim,
        )


@dataclass
class DPOPrefixSharingDataCollatorWithPadding:
    r"""
    Share the prefix
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    pad_to_multiple_of: int = 128

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError
    #     # first, pad everything to the same length
    #     padded_batch = {}
    #     for k in features[0].keys():
    #         if k in ("input_ids", "labels", "position_ids"):
    #             # Set padding value based on the key
    #             if k == "input_ids":
    #                 if self.pad_token_id is None:
    #                     raise ValueError(
    #                         "Padding is enabled, but the tokenizer is not configured with a padding token."
    #                         " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
    #                         " before calling the trainer."
    #                     )
    #                 padding_value = self.pad_token_id
    #             elif k == "labels":
    #                 padding_value = self.label_pad_token_id
    #             elif k == "position_ids":
    #                 padding_value = 1
    #             else:
    #                 raise ValueError(f"Unexpected key in batch '{k}'")

    #             # Convert to tensor and pad
    #             padding_side = "right"
    #             to_pad = [torch.tensor(ex[k], dtype=torch.int64) for ex in features]
    #             padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
    #             if self.pad_to_multiple_of is not None:
    #                 padded_length = padded_batch[k].shape[-1]
    #                 rem = padded_length % self.pad_to_multiple_of
    #                 padded_length += (self.pad_to_multiple_of - rem) % self.pad_to_multiple_of
    #                 padded_batch[k] = pad_to_length(padded_batch[k], padded_length, padding_value)  # pads on the right

    #         elif "logps" in k or "index" in k:
    #             # the cached reference model logprobs
    #             padded_batch[k] = torch.tensor([ex[k] for ex in features])
    #         else:
    #             padded_batch[k] = [ex[k] for ex in features]

    #    return padded_batch


def concat(lists, increment_index=False):
    if increment_index:
        outputs = []
        for l in lists:
            max_index = max(outputs) if len(outputs) > 0 else -1
            outputs.extend([val + max_index + 1 if val != -1 else -1 for val in l])
        return outputs
    return [val for l in lists for val in l]


@dataclass
class DPOPrefixSharingPackedDataCollatorWithPadding:
    r"""
    Share the prefix + add packing
    """

    max_length: int  # required
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    pad_to_multiple_of: int = 128
    padding_mode = "max_length"

    def _combine_samples_to_pack(self, features: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not isinstance(features, list):
            features = [features]
        model_inputs = []
        loss_inputs = []
        for samples_to_pack in features:
            model_inputs_keys = samples_to_pack[0]["model_inputs"].keys()
            out_sample = {key: concat([sample["model_inputs"][key] for sample in samples_to_pack], increment_index=key in ["sequence_id"]) for key in model_inputs_keys}
            out_sample["document_id"] = concat([[i] * len(sample["model_inputs"]["input_ids"]) for i, sample in enumerate(samples_to_pack)], increment_index=False)
            model_inputs.append(out_sample)

            loss_inputs_keys = samples_to_pack[0]["loss_inputs"].keys()
            out_sample = {key: concat([sample["loss_inputs"][key] for sample in samples_to_pack], increment_index=key in ["parent_index"]) for key in loss_inputs_keys}
            loss_inputs.append(out_sample)
        return model_inputs, loss_inputs

    def __call__(self, features: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = dict(
            model_inputs={},
            loss_inputs={}
        )
        model_inputs, loss_inputs = self._combine_samples_to_pack(features)

        for k in model_inputs[0].keys():
            if k in (
                "input_ids",
                "labels",
                "position_ids",
                "sequence_id",
                "preorder_index",
                "postorder_index",
                "document_id"
            ):
                # Set padding value based on the key
                if k == "input_ids":
                    if self.pad_token_id is None:
                        raise ValueError(
                            "Padding is enabled, but the tokenizer is not configured with a padding token."
                            " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                            " before calling the trainer."
                        )
                    padding_value = self.pad_token_id
                elif k == "labels":
                    padding_value = self.label_pad_token_id
                elif k == "position_ids":
                    padding_value = 1
                elif k == "sequence_id":
                    padding_value = -1
                elif k == "document_id":
                    padding_value = -1
                elif "index" in k:
                    padding_value = -1
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                # Convert to tensor and pad
                padding_side = "right"

                to_pad = [torch.tensor(ex[k], dtype=torch.int64) for ex in model_inputs]
                padded_batch["model_inputs"][k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
                max_length = self.max_length
                assert padded_batch["model_inputs"][k].shape[-1] <= max_length
                padded_batch["model_inputs"][k] = pad_to_length(
                    padded_batch["model_inputs"][k], length=max_length, pad_value=padding_value, padding_side=padding_side
                )
                if self.pad_to_multiple_of is not None:
                    padded_length = padded_batch["model_inputs"][k].shape[-1]
                    rem = padded_length % self.pad_to_multiple_of
                    padded_length += (self.pad_to_multiple_of - rem) % self.pad_to_multiple_of
                    padded_batch["model_inputs"][k] = pad_to_length(padded_batch["model_inputs"][k], padded_length, padding_value)  # pads on the right
            elif "logps" in k:
                # the cached reference model logprobs
                padded_batch["model_inputs"][k] = torch.tensor([ex[k] for ex in model_inputs])
            else:
                padded_batch["model_inputs"][k] = [ex[k] for ex in model_inputs]

        # assume bs=1 for now
        assert len(loss_inputs) == 1
        loss_inputs = loss_inputs[0]
        padded_batch["loss_inputs"] = {k: torch.tensor(loss_inputs[k], dtype=torch.int64) for k in loss_inputs}
        # print(padded_batch["loss_inputs"])

        return padded_batch