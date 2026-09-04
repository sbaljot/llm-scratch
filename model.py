"""
model.py
Bundles everything built so far -- embeddings, stacked transformer blocks,
and the output head -- into a single nn.Module. This is just organization:
the forward pass is the exact same chain of calls used throughout demo.py,
just packaged so train.py can treat "the model" as one object with one
set of parameters, instead of juggling loose pieces by hand.
"""

import torch
import torch.nn as nn

from embeddings import TokenAndPositionalEmbedding
from transformer_block import TransformerBlock
from output_head import OutputHead


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = TokenAndPositionalEmbedding(
            vocab_size=vocab_size, max_length=max_length, d_model=d_model, dropout=dropout
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, max_length=max_length, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output_head = OutputHead(d_model=d_model, vocab_size=vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: LongTensor [batch_size, seq_len]
        returns:   FloatTensor [batch_size, seq_len, vocab_size]  (logits)
        """
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x)