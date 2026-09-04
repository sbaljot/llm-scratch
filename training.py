"""
training.py
Optimizer + learning rate schedule setup.

AdamW: the standard optimizer for training transformers. Instead of applying
the same learning rate to every weight update, it keeps a running estimate
of each individual parameter's gradient direction (momentum) and recent
gradient scale (adaptive scaling), so parameters that need bigger or
smaller steps get them automatically. The "W" adds decoupled weight decay --
a small constant pull toward zero applied to every weight, independent of
the gradient -- which helps prevent the model from overfitting by keeping
weights from growing unnecessarily large.

Cosine annealing: rather than training with one fixed learning rate the
whole time, the learning rate follows a smooth curve -- starting at your
chosen peak value and decaying down toward (near) zero, shaped like the
first half of a cosine wave, over the course of training. Big steps early
cover ground quickly while everything is still far from a good solution;
small steps later let the model settle precisely into a good minimum
instead of bouncing past it once it's close.
"""

import torch


def build_optimizer_and_scheduler(
    parameters,
    lr: float = 3e-4,
    total_steps: int = 100,
    weight_decay: float = 0.01,
):
    """
    parameters:   an iterable of model parameters to optimize, e.g.
                  itertools.chain(embed.parameters(), output_head.parameters())
    lr:           peak learning rate -- the highest value the schedule uses,
                  right at step 0
    total_steps:  how many optimizer.step() calls you plan to make in total;
                  the cosine curve is stretched to hit its minimum at exactly
                  this many steps, so this must match your actual training loop
    weight_decay: AdamW's built-in regularization strength (0.01 is a common default)

    Returns (optimizer, scheduler). Call scheduler.step() once per
    optimizer.step() call -- NOT once per epoch -- to follow the intended curve.
    """
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    return optimizer, scheduler