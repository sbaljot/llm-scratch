"""
train.py
The actual training script. Unlike the single-batch overfit sanity-check in
demo.py, this loops over the REAL dataloader -- every batch is different
text, drawn from across your whole corpus -- which is what lets the model
generalize instead of just memorizing one batch.

The four things this script does, every single step, are exactly the four
things you asked for:
  1. Feed a batch of inputs through the model's forward pass -> logits
  2. Compare those logits against the batch's real targets -> loss
  3. loss.backward() -> compute gradients for every weight in the model
  4. optimizer.step() -> update every weight using those gradients

Run:  python train.py
"""

import itertools

import torch

from hf_data import load_hf_text
from dataset import create_dataloader
from model import GPTModel
from loss import compute_loss
from training import build_optimizer_and_scheduler


def get_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2")
    except ImportError:
        from tokenizer import BPETokenizer
        return None  # handled in main() -- needs the corpus text to train on


def main():
    # ---- config -- small on purpose, so this runs in minutes on a single GPU (or CPU for a quick test) ----
    max_length = 32
    stride = 16
    batch_size = 8
    d_model = 64
    num_heads = 4
    num_layers = 2
    num_epochs = 3
    peak_lr = 3e-4
    log_every = 20  # print loss every N steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- data ----
    print("Loading corpus...")
    text = load_hf_text("h0ssn/wikitext-unlearning-mia", split="retain_set")

    tokenizer = get_tokenizer()
    if tokenizer is None:
        from tokenizer import BPETokenizer
        print("tiktoken not installed -- training a from-scratch BPE tokenizer on this corpus...")
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=2000)

    vocab_size = tokenizer.n_vocab if hasattr(tokenizer, "n_vocab") else len(tokenizer)

    dataloader = create_dataloader(
        text, tokenizer, batch_size=batch_size, max_length=max_length, stride=stride
    )
    total_steps = len(dataloader) * num_epochs
    print(f"{len(dataloader)} batches per epoch, {total_steps} total training steps.")

    # ---- model, optimizer, schedule ----
    model = GPTModel(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model.parameters(), lr=peak_lr, total_steps=total_steps
    )

    # ---- training loop ----
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 1. forward pass
            logits = model(inputs)

            # 2. loss against the real next tokens
            loss = compute_loss(logits, targets)

            # 3 & 4. backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if global_step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch} | step {global_step:>5}/{total_steps} "
                    f"| loss {loss.item():.4f} | lr {current_lr:.6f}"
                )

            global_step += 1

    print("\nTraining complete.")

    # ---- save the trained weights AND the config needed to rebuild the model ----
    torch.save(model.state_dict(), "model_weights.pt")

    import json
    config = {
        "vocab_size": vocab_size,
        "max_length": max_length,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
    }
    with open("model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Saved weights to model_weights.pt and config to model_config.json")


if __name__ == "__main__":
    main()