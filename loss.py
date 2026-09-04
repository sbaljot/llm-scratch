"""
loss.py
Cross-entropy loss: turns "how wrong were the model's predictions" into a
single number that training can use to improve the model.

For each position, the model produced a full probability distribution over
50,257 possible tokens (from output_head.py). Cross-entropy compares that
distribution against the ONE token that was actually correct (from
dataset.py's target_ids) and penalizes the model based on how much
probability it assigned to the right answer:

  - if the model gave the correct token high probability -> low loss (good)
  - if the model gave the correct token low probability  -> high loss (bad)

Specifically: loss = -log(probability the model assigned to the correct token).
A perfect prediction (probability 1.0 on the right token) gives loss = 0.
A confidently WRONG prediction (probability near 0 on the right token) gives
a loss that shoots toward infinity -- this is what makes cross-entropy punish
confident mistakes much harder than uncertain ones.
"""

import math

import torch
import torch.nn as nn


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  FloatTensor [batch_size, seq_len, vocab_size]  -- raw scores from
             output_head.py, BEFORE softmax (nn.CrossEntropyLoss applies
             softmax internally, in a more numerically stable way than doing
             it yourself first)
    targets: LongTensor  [batch_size, seq_len]               -- the real next
             token ids, straight from dataset.py's target_ids

    returns: a single scalar tensor -- the average loss across every
             position in the batch
    """
    batch_size, seq_len, vocab_size = logits.shape

    # nn.CrossEntropyLoss expects logits shaped [N, num_classes] and targets
    # shaped [N] -- so flatten the batch and sequence dimensions together,
    # treating every (sequence, position) pair as one independent prediction
    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    targets_flat = targets.view(batch_size * seq_len)

    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits_flat, targets_flat)


def random_baseline_loss(vocab_size: int) -> float:
    """
    The loss you'd expect from a completely untrained model that assigns
    EQUAL probability to every token (1 / vocab_size each). Useful as a
    sanity-check reference point: a freshly initialized model's loss should
    land close to this value. If training is working, loss should steadily
    drop below it.
    """
    return math.log(vocab_size)