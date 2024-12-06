from torch.nn.attention.flex_attention import create_block_mask, and_masks


def get_causal_mask():
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    return causal


def get_tree_mask(preorder_index, postorder_index):
    def tree(b, h, q_idx, kv_idx):
        preorder_ind_q = preorder_index[b, q_idx]
        postorder_ind_q = postorder_index[b, q_idx]
        preorder_ind_kv = preorder_index[b, kv_idx]
        postorder_ind_kv = postorder_index[b, kv_idx]
        return (preorder_ind_kv <= preorder_ind_q) & (postorder_ind_kv >= postorder_ind_q)
    return tree


def get_packing_mask(document_id):
    def packing(b, h, q_idx, kv_idx):
        return document_id[b, q_idx] == document_id[b, kv_idx]
    return packing


def construct_flex_mask(masks, seq_len, compile=False):
    block_mask = create_block_mask(
        and_masks(*masks), B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile
    )
    return block_mask


def construct_causal_mask(seq_len, compile=False):
    """Simple causal block mask for causal LMs"""
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile
    )
    return block_mask

def construct_dpo_mask(chosen_index, rejected_index, seq_len, compile=False):
    """Block-sparse mask for prefix shared inputs"""
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
    """Block-sparse mask for packed prefix shared inputs"""
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
        # no longer needed because seq_id padding takes care of this
        # padding_mask = (kv_idx < end_index[b][kv_idx])
        return dpo_prefix_sharing_mask & sequence_mask

    block_mask = create_block_mask(
        document_causal_mask, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile
    )
    return block_mask