from torch.nn.attention.flex_attention import create_block_mask


def construct_causal_mask(seq_len, compile=False):
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile
    )
    return block_mask


def construct_dpo_mask(chosen_index, rejected_index, seq_len, compile=False):
    def dpo_mask(b, h, q_idx, kv_idx):
        return (~((q_idx >= rejected_index[b]) & (chosen_index[b] <= kv_idx) & (kv_idx < rejected_index[b]))) & (
            q_idx >= kv_idx
        )

    block_mask = create_block_mask(
        dpo_mask, B=len(chosen_index), H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile
    )
    return block_mask

def construct_dpo_mask_with_packing(
    sequence_id, chosen_index, rejected_index, end_index, batch_size, seq_len, index_seq_len, compile=False
):
    def document_causal_mask(b, h, q_idx, kv_idx):
        seq_idx = sequence_id[b * index_seq_len + q_idx]
        dpo_prefix_sharing_mask = (
            ~(
                (q_idx >= rejected_index[b * index_seq_len + q_idx])
                & (chosen_index[b * index_seq_len + q_idx] <= kv_idx)
                & (kv_idx < rejected_index[b * index_seq_len + q_idx])
            )
        ) & (q_idx >= kv_idx)
        sequence_mask = seq_idx == sequence_id[b * index_seq_len + kv_idx]
        # padding_mask = (kv_idx < end_index[b][kv_idx])
        return dpo_prefix_sharing_mask & sequence_mask

    block_mask = create_block_mask(
        document_causal_mask, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile
    )
    return block_mask