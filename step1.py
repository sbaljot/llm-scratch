"""
demo.py
End-to-end pipeline: Hugging Face dataset -> tokenizer -> Dataset -> DataLoader -> batch.

Data source: h0ssn/wikitext-unlearning-mia (Hugging Face)
Run:  python demo.py

Note: this dataset appears to be a membership-inference-attack (MIA) / unlearning
benchmark rather than a large pretraining corpus (based on sibling datasets from
the same author). It's likely a small set of short text passages, not millions of
tokens of running prose. This script will load whatever text it can find and
report the size -- if it's too small for GPTDataset's window size, lower
max_length or point load_hf_text at a bigger corpus (see hf_data.py's __main__
example for wikitext-103 instead).
"""

from hf_data import load_hf_text
from dataset import create_dataloader


def get_tokenizer():
    try:
        import tiktoken

        print("Using tiktoken's GPT-2 BPE tokenizer (vocab size 50257).")
        return tiktoken.get_encoding("gpt2")
    except ImportError:
        print(
            "tiktoken not installed -- falling back to the from-scratch "
            "BPETokenizer (run: pip install tiktoken, to use the real GPT-2 vocab)."
        )
        from tokenizer import BPETokenizer

        # need some text to train the from-scratch tokenizer on; reuse the corpus itself
        tok = BPETokenizer()
        return tok  # trained below, once we have the corpus text


def main():
    print("Loading h0ssn/wikitext-unlearning-mia from Hugging Face...")
    text = load_hf_text(
        "h0ssn/wikitext-unlearning-mia",
        split="retain_set",
    )

    tokenizer = get_tokenizer()
    from tokenizer import BPETokenizer  # local import to check type without circular clutter

    if isinstance(tokenizer, BPETokenizer):
        print("Training from-scratch BPE tokenizer on the loaded corpus...")
        tokenizer.train(text, vocab_size=1000)

    max_length = 32
    stride = 16

    dataloader = create_dataloader(
        text,
        tokenizer,
        batch_size=4,
        max_length=max_length,
        stride=stride,
    )

    inputs, targets = next(iter(dataloader))
    print("\nInput batch shape: ", inputs.shape)   # [batch_size, max_length]
    print("Target batch shape:", targets.shape)    # [batch_size, max_length]
    print("\nFirst input sequence (token ids): ", inputs[0].tolist())
    print("First target sequence (token ids):", targets[0].tolist())
    print(
        "\nNotice target[0] == input[1], target[1] == input[2], etc. "
        "-- that's the 'shifted by one' next-token setup."
    )


if __name__ == "__main__":
    main()