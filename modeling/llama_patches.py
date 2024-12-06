from typing import Dict, Optional, Union, Tuple

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaModel,
    LLAMA_ATTENTION_CLASSES,
    apply_rotary_pos_emb,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

class LlamaFlexAttention(LlamaAttention):
    """
    Modified version of LlamaFlashAttention2 to use FlexAttention
    Adapted from https://github.com/huggingface/transformers/blob/37ea04013b34b39c01b51aeaacd8d56f2c62a7eb/src/transformers/models/llama/modeling_llama.py#L399
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs
    ) -> Tuple[torch.Tensor, None, None]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        assert self.attention_dropout == 0 # flex attention does not directly support dropout yet

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if attention_mask is None:
            raise RuntimeError("No attention mask specified")

        assert q_len % 128 == 0, f"flex_attention requires seq len to be divisble by 128, but got seq of length {q_len}"
        assert attention_mask.shape == (bsz, 1, q_len, q_len) or attention_mask.shape == (1, 1, q_len, q_len), f"got attention_mask of shape {attention_mask.shape}, expected {(bsz, 1, q_len, q_len)} or {(1, 1, q_len, q_len)}"

        attn_output = self.flex_attention_compiled(
            query_states, key_states, value_states, block_mask=attention_mask, enable_gqa=True
        ).transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, None


def patch_flash_attention(config):
    LLAMA_ATTENTION_CLASSES["flex_attention"] = LlamaFlexAttention
    # due to how python's name mangling works, you can't directly override a private method with subclassing without also override the public methods where it's used
    LlamaForCausalLM._update_causal_mask = LlamaForCausalLMFlexAttn._update_causal_mask
    LlamaModel._update_causal_mask = LlamaForCausalLMFlexAttn._update_causal_mask
    LlamaForCausalLM._autoset_attn_implementation = LlamaForCausalLMFlexAttn._autoset_attn_implementation
    LlamaModel._autoset_attn_implementation = LlamaForCausalLMFlexAttn._autoset_attn_implementation
    config._attn_implementation = "flex_attention"
    return config


class LlamaForCausalLMFlexAttn(LlamaForCausalLM):
    def __init__(self, config):
        config = patch_flash_attention(config)
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
