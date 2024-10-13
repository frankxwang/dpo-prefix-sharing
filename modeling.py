from typing import Dict, Optional, Union

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import *
from transformers.utils import (
    is_torch_xla_available,
    logging,
)


logger = logging.get_logger(__name__)


class MistralFlexAttention(MistralAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # this will be set during the model init so that we don't need to recompile multiple times
        self.flex_attention_compiled = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[BlockMask] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        assert self.attention_dropout == 0  # dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if attention_mask is None:
            raise RuntimeError("No mask specified")

        assert q_len % 128 == 0, f"flex_attention requires seq len to be divisble by 128, but got seq of length {q_len}"
        assert attention_mask.shape == (bsz, 1, q_len, q_len) or attention_mask.shape == (1, 1, q_len, q_len), f"got attention_mask of shape {attention_mask.shape}, expected {(bsz, 1, q_len, q_len)} or {(1, 1, q_len, q_len)}"

        attn_output = self.flex_attention_compiled(
            query_states, key_states, value_states, block_mask=attention_mask
        ).permute(0, 2, 1, 3)

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, None


def hijack_flash_attention(config):
    MISTRAL_ATTENTION_CLASSES["flex_attention"] = MistralFlexAttention
    # due to how python's name mangling works, you can't directly override a private method with subclassing without also override the public methods where it's used
    MistralForCausalLM._update_causal_mask = MistralForCausalLMFlexAttn._update_causal_mask
    MistralModel._update_causal_mask = MistralForCausalLMFlexAttn._update_causal_mask
    MistralForCausalLM._autoset_attn_implementation = MistralForCausalLMFlexAttn._autoset_attn_implementation
    MistralModel._autoset_attn_implementation = MistralForCausalLMFlexAttn._autoset_attn_implementation
    config._attn_implementation = "flex_attention"
    return config


class MistralForCausalLMFlexAttn(MistralForCausalLM):
    def __init__(self, config):
        config = hijack_flash_attention(config)
        super().__init__(config)

        # set flex_attention_compiled for all layers
        self._set_flex_attention_compiled()
    
    def _set_flex_attention_compiled(self):
        # explicit backend for clarity
        compiled_flex_attn = torch.compile(flex_attention, backend="inductor")
        for layer in self.model.layers:
            layer.self_attn.flex_attention_compiled = compiled_flex_attn

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        use_cache: bool,
        output_attentions: bool,
    ):
        return attention_mask

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        return config


def construct_dpo_mask(chosen_index, rejected_index, seq_len, compile=False):
    def dpo_mask(b, h, q_idx, kv_idx):
        return (~((q_idx >= rejected_index[b]) & (chosen_index[b] <= kv_idx) & (kv_idx < rejected_index[b]))) & (
            q_idx >= kv_idx
        )

    block_mask = create_block_mask(
        dpo_mask, B=len(chosen_index), H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile
    )
    return block_mask


def construct_causal_mask(seq_len):
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=False
    )
    return block_mask

def construct_dpo_mask_with_packing_old(sequence_id_flattened, chosen_index, rejected_index, end_index, seq_len, batch_size):
    def document_causal_mask(b, h, q_idx, kv_idx):
        seq_idx =  sequence_id_flattened[b*seq_len + q_idx]
        dpo_prefix_sharing_mask = (~((q_idx >= rejected_index[seq_idx]) & (chosen_index[seq_idx] <= kv_idx) & (kv_idx < rejected_index[seq_idx]))) & (
            q_idx >= kv_idx
        )
        sequence_mask = seq_idx == sequence_id_flattened[b*seq_len + kv_idx]
        # padding_mask = (kv_idx < end_index[b][kv_idx])
        return dpo_prefix_sharing_mask & sequence_mask
    
    block_mask = create_block_mask(document_causal_mask, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=False)
    return block_mask

# All flattened
def construct_dpo_mask_with_packing(sequence_id, chosen_index, rejected_index, end_index, batch_size, seq_len, index_seq_len, compile=False):
    def document_causal_mask(b, h, q_idx, kv_idx):
        seq_idx =  sequence_id[b*index_seq_len + q_idx]
        dpo_prefix_sharing_mask = (~((q_idx >= rejected_index[b*index_seq_len + q_idx]) & (chosen_index[b*index_seq_len + q_idx] <= kv_idx) & (kv_idx < rejected_index[b*index_seq_len + q_idx]))) & (
            q_idx >= kv_idx
        )
        sequence_mask = seq_idx == sequence_id[b*index_seq_len + kv_idx]
        # padding_mask = (kv_idx < end_index[b][kv_idx])
        return dpo_prefix_sharing_mask & sequence_mask
    
    block_mask = create_block_mask(document_causal_mask, B=batch_size, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=compile)
    return block_mask