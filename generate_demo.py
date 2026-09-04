"""
generate_demo.py
Loads your trained model and generates text from a prompt, using generate.py.

Run:  python generate_demo.py
"""

import json
import os

import torch

from model import GPTModel
from generate import generate


def get_tokenizer(vocab_size: int):
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2")
    except ImportError:
        raise RuntimeError(
            "This demo expects tiktoken (matching what train.py used). "
            "If you trained with the from-scratch BPETokenizer instead, "
            "you'll need to save/reload that tokenizer's merges too -- "
            "ask if you want that wired up."
        )


def main():
    # ---- load config (falls back to train.py's defaults if not found, e.g.
    # if your training run finished before this config-saving code was added) ----
    if os.path.exists("model_config.json"):
        with open("model_config.json") as f:
            config = json.load(f)
    else:
        print("model_config.json not found -- falling back to train.py's defaults.")
        config = {"vocab_size": 50257, "max_length": 32, "d_model": 64, "num_heads": 4, "num_layers": 2}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- rebuild the exact same architecture, then load the trained weights into it ----
    model = GPTModel(
        vocab_size=config["vocab_size"],
        max_length=config["max_length"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
    ).to(device)

    model.load_state_dict(torch.load("model_weights.pt", map_location=device))
    model.eval()

    tokenizer = get_tokenizer(config["vocab_size"])

    prompt = "The history of"
    print(f"Prompt: {prompt!r}\n")

    output = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=40,
        max_length=config["max_length"],
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        device=device,
    )
    print("Generated:\n", output)


if __name__ == "__main__":
    main()