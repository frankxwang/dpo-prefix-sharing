from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import torch
from trl.trainer.utils import pad, pad_to_length


@dataclass
class DPOPrefixSharingDataCollatorWithPadding:
    r"""
    Share the prefix
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    pad_to_multiple_of: int = 128

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k in ("input_ids", "labels", "position_ids"):
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
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                # Convert to tensor and pad
                padding_side = "right"
                to_pad = [torch.tensor(ex[k], dtype=torch.int64) for ex in features]
                padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
                if self.pad_to_multiple_of is not None:
                    padded_length = padded_batch[k].shape[-1]
                    rem = padded_length % self.pad_to_multiple_of
                    padded_length += (self.pad_to_multiple_of - rem) % self.pad_to_multiple_of
                    padded_batch[k] = pad_to_length(padded_batch[k], padded_length, padding_value)  # pads on the right

            elif "logps" in k or "index" in k:
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


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
        out_features = []
        for samples_to_pack in features:
            out_sample = {}
            all_keys = list(samples_to_pack[0].keys())
            permitted_keys = [
                "input_ids",
                "labels",
                "position_ids",
                "chosen_index",
                "rejected_index",
                "end_index",
                "sequence_id",
            ]
            permitted_keys.extend([key for key in all_keys if "logps" in key])
            for sample_ind, sample in enumerate(samples_to_pack):
                if not out_sample:
                    out_sample = {key: [] for key in sample if key in permitted_keys}
                    out_sample["loss_seq_id"] = []
                sample["loss_seq_id"] = [2 * sample_ind + 1] * (sample["rejected_index"][0]) + [2 * sample_ind + 2] * (
                    sample["length"] - sample["rejected_index"][0]
                )
                for key in list(out_sample.keys()):
                    if "index" in key:
                        sample[key] = list(np.array(sample[key]) + len(out_sample[key]))
                    out_sample[key] = out_sample[key] + sample[key]
                assert len(sample["loss_seq_id"]) == len(sample["input_ids"])
            out_features.append(out_sample)
        return out_features

    def __call__(self, features: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        packed_features: List[Dict[str, Any]] = self._combine_samples_to_pack(features)

        for k in packed_features[0].keys():
            if k in (
                "input_ids",
                "labels",
                "position_ids",
                "sequence_id",
                "chosen_index",
                "rejected_index",
                "end_index",
                "loss_seq_id",
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
                elif k == "loss_seq_id":
                    padding_value = 0
                elif "index" in k:
                    padding_value = -1
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                # Convert to tensor and pad
                padding_side = "right"

                to_pad = [torch.tensor(ex[k], dtype=torch.int64) for ex in packed_features]
                padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
                max_length = self.max_length
                assert padded_batch[k].shape[-1] <= max_length
                padded_batch[k] = pad_to_length(
                    padded_batch[k], length=max_length, pad_value=padding_value, padding_side=padding_side
                )
                if self.pad_to_multiple_of is not None:
                    padded_length = padded_batch[k].shape[-1]
                    rem = padded_length % self.pad_to_multiple_of
                    padded_length += (self.pad_to_multiple_of - rem) % self.pad_to_multiple_of
                    padded_batch[k] = pad_to_length(padded_batch[k], padded_length, padding_value)  # pads on the right
                if k == "loss_seq_id":
                    for i in range(1, len(padded_batch[k])):
                        padded_batch[k][i][padded_batch[k][i] != 0] += torch.max(padded_batch[k][i - 1])
            elif "logps" in k:
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in packed_features])
            else:
                padded_batch[k] = [ex[k] for ex in packed_features]

        return padded_batch