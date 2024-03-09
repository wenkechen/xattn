from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN

from sequences.models.gpt2.config import GPT2Config


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx=None) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale_attn_weights = config.scale_attn_weights
        self.softmax_scale = self.head_dim ** (-0.5) if self.scale_attn_weights else 1.0

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        self.c_attn = nn.Conv1D(3 * self.embed_dim, self.embed_dim)
        self.attn = SelfAttention(attn_drop=config.attn_pdrop, softmax_scale=self.softmax_scale)
        self.c_proj = nn.Conv1D(self.embed_dim, self.embed_dim)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_state: Optional[torch.FloatTensor],
        residul: Optional[torch.FloatTensor],
        cu_seqlens: Optional[torch.IntTensor],
        max_seqlen: Optional[int],
    ):
        query, key, value = self.c_attn(hidden_state).split(self.split_size, dim=-1)
        query = query.view(-1, query.shape[-1] // self.head_dim, self.head_dim)
        key = key.view(-1, key.shape[-1] // self.head_dim, self.head_dim)
        value = value.view(-1, value.shape[-1] // self.head_dim, self.head_dim)

        attn_output = self.attn(query, key, value, cu_seqlens, max_seqlen, causal=True)
        attn_output = attn_output.contiguous().view(attn_output.shape[0], -1)

        attn_output = self.c_proj(attn_output, residul)
        attn_output = self.resid_dropout(attn_output)
        if residul:
            attn_output += residul

        return attn_output


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size: int, config: GPT2Config, layer_idx=None):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Conv1D(intermediate_size, embed_dim)
        self.c_proj = nn.Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]  # Fast Gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_state: Optional[torch.FloatTensor],
        residul: Optional[torch.FloatTensor],
    ):
        hidden_state = self.c_fc(hidden_state)
        hidden_state = self.act(hidden_state)

        hidden_state = self.c_proj(hidden_state)
        hidden_state = self.dropout(hidden_state)
        if residul:
            hidden_state += residul

        return hidden_state


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config, layer_idx)

    def forward(
        self,
        hidden_state: Optional[torch.FloatTensor],
        cu_seqlens: Optional[torch.IntTensor],
        max_seqlen: Optional[int],
    ):
        residual = hidden_state
        hidden_state = self.ln_1(hidden_state)
        attn_output = self.attn(hidden_state, residual, cu_seqlens, max_seqlen)

        hidden_state = attn_output
        residual = hidden_state
        hidden_state = self.ln_2(hidden_state)
        hidden_state = self.mlp(hidden_state, residual)

        return hidden_state
