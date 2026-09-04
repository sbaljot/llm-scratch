"""
output_head.py
The final layer: maps each token's hidden state (still just a d_model-sized
vector, the same shape that's been flowing through every block) into a
prediction over the ENTIRE vocabulary -- "given everything seen so far,
how likely is each possible next token?"

Input:  hidden_states, shape [batch_size, seq_len, d_model]
Output: logits,        shape [batch_size, seq_len, vocab_size]

One number per possible token, at every position, for every sequence in
the batch. Turning those numbers into actual probabilities (softmax) or
sampling an actual next token is a separate step -- this layer's only job
is producing the raw scores (logits).
"""

import torch
import torch.nn as nn


class OutputHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        d_model:    hidden dimension coming out of the last transformer block
        vocab_size: number of possible tokens (must match the tokenizer's
                    vocab_size used in embeddings.py -- e.g. 50257 for
                    tiktoken's gpt2 encoding)
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: FloatTensor [batch_size, seq_len, d_model]
        returns:       FloatTensor [batch_size, seq_len, vocab_size]  (logits,
                       NOT yet probabilities -- apply softmax separately if
                       you need an actual distribution, e.g. for sampling)
        """
        return self.proj(hidden_states)