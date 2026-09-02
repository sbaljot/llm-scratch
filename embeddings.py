"""
embeddings.py
The first layer of the model: turns integer token IDs into dense vectors
the network can actually do math on.

Two lookup tables, added together:
  1. Token embedding    -- "what token is this?"      (vocab_size x d_model)
  2. Positional embedding -- "where in the sequence is it?" (max_length x d_model)

Input:  token_ids, shape [batch_size, seq_len]        (integers)
Output: embeddings,  shape [batch_size, seq_len, d_model] (floats)
"""

import torch
import torch.nn as nn


class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, d_model: int, dropout: float = 0.1):
        """
        vocab_size: number of unique tokens the tokenizer can produce
                    (e.g. 50257 for tiktoken's gpt2 encoding, or len(tokenizer)
                    for the from-scratch BPETokenizer)
        max_length: the context window size (must be >= any seq_len you'll pass in;
                    should match max_length in dataset.py's create_dataloader)
        d_model:    the embedding dimension -- how many numbers represent each token
                    (e.g. 768 for GPT-2 small; 64-128 is plenty for a first small model)
        dropout:    regularization applied after combining the two embeddings
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: LongTensor of shape [batch_size, seq_len]
        returns:   FloatTensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = token_ids.shape

        # look up "what" each token is
        tok_emb = self.token_embedding(token_ids)  # [batch_size, seq_len, d_model]

        # look up "where" each position is: [0, 1, 2, ..., seq_len-1], broadcast over the batch
        positions = torch.arange(seq_len, device=token_ids.device)  # [seq_len]
        pos_emb = self.position_embedding(positions)                # [seq_len, d_model]

        # combine: every token's vector now encodes both identity and position
        x = tok_emb + pos_emb  # broadcasts pos_emb across the batch dimension
        return self.dropout(x)