"""
attention.py
Causal self-attention: the mechanism that lets each token's vector gather
context from the tokens before it (and only before it).

Input:  x, shape [batch_size, seq_len, d_model]   (output of embeddings.py)
Output:   shape [batch_size, seq_len, d_model]     (context-aware vectors)

Steps, per token:
  1. Project x into Query (Q), Key (K), Value (V) -- three learned linear views
     of the same vector, each asking a different question:
       Q: "what am I looking for?"
       K: "what do I contain, that others might look for?"
       V: "what do I actually offer, once someone attends to me?"
  2. Score every token's Query against every token's Key (dot product) -- higher
     score means "more relevant to me".
  3. Apply the causal mask: zero out (set to -inf before softmax) any score
     where a token would look at a position AFTER itself. This is what makes
     it valid for next-token prediction -- token i can only see tokens 0..i.
  4. Softmax the scores into weights (they sum to 1 per token), then use those
     weights to blend the Value vectors -- each token's new representation is
     a weighted mix of the tokens it's allowed to see.
"""

import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, max_length: int, dropout: float = 0.1):
        """
        d_model:    embedding dimension (must match embeddings.py's d_model)
        max_length: the largest sequence length this layer will ever see;
                    used to pre-build the causal mask once, up front
        dropout:    applied to the attention weights (a common regularization
                    trick -- randomly zeroes some attention connections during
                    training so the model doesn't over-rely on any one link)
        """
        super().__init__()
        self.d_model = d_model

        # three separate learned projections -- same input, three different views
        self.W_query = nn.Linear(d_model, d_model, bias=False)
        self.W_key = nn.Linear(d_model, d_model, bias=False)
        self.W_value = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # the causal mask: a [max_length, max_length] upper-triangular matrix
        # of 1s above the diagonal (future positions) and 0s elsewhere.
        # registered as a buffer, not a parameter -- it's fixed, never learned,
        # but still moves with the model when you call .to(device).
        causal_mask = torch.triu(torch.ones(max_length, max_length), diagonal=1)
        self.register_buffer("causal_mask", causal_mask.bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        queries = self.W_query(x)  # [batch_size, seq_len, d_model]
        keys = self.W_key(x)       # [batch_size, seq_len, d_model]
        values = self.W_value(x)   # [batch_size, seq_len, d_model]

        # raw attention scores: every query dotted with every key
        # [batch_size, seq_len, seq_len] -- scores[b, i, j] = how much token i
        # (query) attends to token j (key)
        scores = queries @ keys.transpose(-2, -1)

        # scale down -- without this, scores grow with d_model and softmax
        # saturates (gradients vanish). Standard "scaled dot-product attention".
        scores = scores / math.sqrt(d_model)

        # apply the causal mask: wherever mask is True (j > i, a future token),
        # set the score to -inf so softmax turns it into ~0 attention weight
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask, float("-inf"))

        # turn scores into a proper probability distribution per token (each
        # row sums to 1) -- these are the actual attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # blend the Value vectors according to those weights
        context = attn_weights @ values  # [batch_size, seq_len, d_model]

        return context