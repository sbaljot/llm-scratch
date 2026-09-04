"""
sampling.py
The controls that shape HOW a token gets picked from the model's predicted
distribution -- these don't change what the model "thinks" (the logits),
they change how those logits get turned into an actual choice.

Without any of these, sampling straight from the raw softmax distribution
tends to occasionally pick from the long tail of barely-plausible tokens,
which is a common cause of generated text going incoherent a few tokens
later. All three controls exist to trim or reshape that distribution before
sampling, trading off randomness against reliability.
"""

from typing import Optional

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Rescales logits before softmax.
      temperature < 1.0: sharpens the distribution -- the model's favorite
        tokens get relatively MORE likely, unlikely tokens get pushed toward
        zero. temperature -> 0 approaches always picking the single most
        likely token (deterministic, but often repetitive/boring).
      temperature > 1.0: flattens the distribution -- probabilities move
        closer to uniform, so more unusual tokens get a real chance of being
        picked (more "creative", but risks incoherence if pushed too high).
      temperature == 1.0: no change, use the model's raw distribution as-is.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keeps only the k highest-scoring tokens; every other token's logit gets
    set to -inf so it becomes impossible to sample after softmax.

    Fixed-size cutoff: always keeps exactly k candidates, regardless of how
    confident or uncertain the model actually is at this step. That's its
    main weakness -- see apply_top_p below for the adaptive alternative.
    """
    top_values, _ = torch.topk(logits, k)
    threshold = top_values[..., -1, None]  # the k-th highest value, kept as the cutoff
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    "Nucleus sampling": keeps the SMALLEST set of highest-probability tokens
    whose cumulative probability adds up to at least p, discards the rest.

    Unlike top-k's fixed count, this adapts to the model's confidence at
    each step: when the model is very sure (probability concentrated on a
    few tokens), the kept set shrinks automatically; when it's uncertain
    (probability spread thin across many tokens), the kept set grows to
    include more options. This usually gives more consistently coherent
    output than a fixed top-k across varying contexts.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # find the cutoff point: the first position where cumulative probability
    # exceeds p -- keep everything up to and including it
    sorted_mask = cumulative_probs > p
    # shift right by one so the token that CROSSES the threshold is still kept
    # (otherwise we'd sometimes end up with an empty set)
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # scatter back into the original (unsorted) token order
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(-1, sorted_indices, sorted_logits)
    return filtered_logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Applies temperature, then top-k, then top-p (in that order -- each
    narrows the field further), and samples one token id from what remains.

    logits: FloatTensor [vocab_size] -- raw scores for ONE position
            (e.g. output_head's output at the last sequence position)
    returns: LongTensor [1] -- the sampled token id
    """
    logits = apply_temperature(logits, temperature)

    if top_k is not None:
        logits = apply_top_k(logits, top_k)

    if top_p is not None:
        logits = apply_top_p(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)