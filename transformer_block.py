"""
transformer_block.py
Combines multi-head causal self-attention with a feed-forward network,
layer normalization, and residual connections into one reusable
"transformer block" -- the modular unit that gets stacked N times deep
to build the full model.

Architecture per block (GPT-2 style, "pre-norm"):
    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

Input/output shape: [batch_size, seq_len, d_model] -- UNCHANGED, which is
exactly what lets you stack these blocks directly on top of each other.
"""

import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Same causal self-attention idea as attention.py's CausalSelfAttention,
    generalized to run several smaller attention "heads" in parallel instead
    of one big one. Each head gets its own slice of d_model to work with, so
    different heads can end up specializing in different kinds of relationships
    (e.g. one head might track nearby words, another might track subject/verb
    agreement across a long sentence) -- something a single head can't do as
    easily since it has to represent everything in one shared space.
    """

    def __init__(self, d_model: int, num_heads: int, max_length: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # how many dims each head gets
        self.d_model = d_model

        self.W_query = nn.Linear(d_model, d_model, bias=False)
        self.W_key = nn.Linear(d_model, d_model, bias=False)
        self.W_value = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)  # recombines heads back into one vector

        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(torch.ones(max_length, max_length), diagonal=1)
        self.register_buffer("causal_mask", causal_mask.bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # split d_model into (num_heads, head_dim), then move heads into their
        # own dimension so each head's attention is computed independently
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # shapes now: [batch_size, num_heads, seq_len, head_dim]

        scores = queries @ keys.transpose(-2, -1)  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ values  # [batch_size, num_heads, seq_len, head_dim]

        # merge heads back into a single d_model-sized vector per token
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.out_proj(context)


class FeedForward(nn.Module):
    """
    A simple two-layer MLP applied independently to each token's vector.
    Attention mixes information ACROSS tokens; the FFN then processes each
    token's (now context-aware) vector on its own, adding representational
    capacity. The expansion (4x) and back-down pattern is the standard
    transformer design (from the original "Attention Is All You Need" paper).
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One full transformer block: attention sub-layer + FFN sub-layer, each
    wrapped in a residual connection and preceded by layer normalization.

    Residual connections (the "x = x + ..." pattern) are what make it
    possible to stack many of these blocks deep without gradients vanishing:
    even if a sub-layer's output were useless, the input x still flows
    straight through unchanged via the "+", so gradients always have a clear
    path back to earlier layers during backpropagation.

    Layer normalization rescales each token's vector (zero mean, unit
    variance, then a learned scale/shift) before it enters a sub-layer --
    this keeps values in a stable range as they pass through many stacked
    blocks, which is what makes deep stacks trainable at all.
    """

    def __init__(self, d_model: int, num_heads: int, max_length: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, max_length, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))  # residual around attention
        x = x + self.ff(self.norm2(x))          # residual around the FFN
        return x
