"""
generate.py
Autoregressive text generation: given a starting prompt, repeatedly predict
and append one token at a time, feeding the model's own growing output back
in as its next input -- that feedback loop is what "autoregressive" means.

Sampling controls (temperature, top-k, top-p) live in sampling.py and get
applied here at the one point that matters: right before a token is chosen.
"""

from typing import Optional

import torch

from sampling import sample_token


@torch.no_grad()  # generation doesn't need gradients -- saves memory and time
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    max_length: int = 32,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device=None,
) -> str:
    """
    model:          a GPTModel instance (ideally trained; put it in .eval() mode --
                     this function does that for you)
    tokenizer:      the SAME tokenizer used during training (vocab_size must match)
    prompt:         the starting text, e.g. "Once upon a time"
    max_new_tokens: how many new tokens to generate after the prompt
    max_length:     the model's context window (must match its training config).
                     If the running sequence grows past this, it gets cropped to
                     the most recent max_length tokens before every forward pass.
    temperature:    > 1.0 = more random/creative, < 1.0 = more confident/repetitive,
                     1.0 = unchanged. See sampling.py for the full explanation.
    top_k:          keep only the k highest-probability tokens each step.
                     None = no top-k filtering.
    top_p:          nucleus sampling -- keep the smallest set of tokens whose
                     cumulative probability reaches p (e.g. 0.9). None = no
                     top-p filtering. Can be combined with top_k (top_k narrows
                     first, then top_p narrows further within what's left).

    Returns the full generated text (prompt + newly generated tokens, decoded).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]

    for _ in range(max_new_tokens):
        # crop to the model's context window before feeding it in
        context = token_ids[:, -max_length:]

        # forward pass -> logits for every position
        logits = model(context)  # [1, seq_len, vocab_size]

        # only the prediction for what comes AFTER the last token matters
        next_token_logits = logits[0, -1, :]  # [vocab_size]

        # temperature + top-k + top-p, then sample one token id
        next_token_id = sample_token(
            next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p
        )  # [1]

        # append the new token and loop -- next iteration sees this token too
        token_ids = torch.cat([token_ids, next_token_id.unsqueeze(0)], dim=1)

    generated_ids = token_ids[0].tolist()
    return tokenizer.decode(generated_ids)