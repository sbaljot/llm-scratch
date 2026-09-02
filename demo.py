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
from embeddings import TokenAndPositionalEmbedding
from attention import CausalSelfAttention
from transformer_block import TransformerBlock
import torch.nn as nn


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
    # NOTE: this dataset has no "train" split -- it's organized as MIA/unlearning
    # benchmark splits instead (e.g. "retain_set", likely also a "forget_set" /
    # "eval_set" or similar). Using "retain_set" here since that's the data this
    # kind of benchmark expects a model to have been trained on.
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

    # --- Step 2 starts here: turn those token IDs into vectors ---
    vocab_size = tokenizer.n_vocab if hasattr(tokenizer, "n_vocab") else len(tokenizer)
    d_model = 64  # small on purpose -- easy to run on CPU while you're testing shapes

    embed = TokenAndPositionalEmbedding(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
    )

    embedded = embed(inputs)  # inputs: [batch_size, max_length] -> [batch_size, max_length, d_model]
    print("\nEmbedding output shape:", embedded.shape)
    print("First token's embedding vector (first 8 of", d_model, "dims):")
    print(embedded[0, 0, :8])

    # --- causal self-attention: let tokens gather context from earlier tokens ---
    attention = CausalSelfAttention(d_model=d_model, max_length=max_length)
    attention.eval()  # disable dropout for this exploratory run -- otherwise
    # token 0 (which has only ONE valid attention weight, since the causal
    # mask blocks everything else) has a real chance of that single weight
    # being dropped, zeroing its entire output. eval() makes runs reproducible
    # while you're just checking shapes; re-enable dropout (attention.train())
    # once you move on to actual training.
    context = attention(embedded)  # [batch_size, max_length, d_model] -> same shape out

    print("\nAttention output shape:", context.shape)
    print("First token's context vector (first 8 of", d_model, "dims):")
    print(context[0, 0, :8])
    print(
        "\nNote: the first token's context vector should differ from its raw "
        "embedding above -- even though it only had itself to attend to "
        "(causal mask blocks everything after position 0), it still passed "
        "through the Q/K/V projections."
    )

    # --- stack transformer blocks: multi-head attention + FFN + residuals + norm ---
    num_heads = 4       # d_model (64) must be divisible by this -> head_dim = 16
    num_layers = 2       # how many blocks deep -- GPT-2 small uses 12

    blocks = nn.ModuleList([
        TransformerBlock(d_model=d_model, num_heads=num_heads, max_length=max_length)
        for _ in range(num_layers)
    ])
    for block in blocks:
        block.eval()  # disable dropout for this exploratory, reproducible run

    x = embedded  # start from the embedding output, same input the single-head demo used
    for block in blocks:
        x = block(x)  # shape stays [batch_size, max_length, d_model] at every layer

    print("\nAfter", num_layers, "transformer block(s), output shape:", x.shape)
    print("First token's vector (first 8 of", d_model, "dims):")
    print(x[0, 0, :8])
    print(
        "\nUnlike the single-head attention demo, token 0's vector here is no "
        "longer just its own Value projection -- the residual connections mean "
        "its original embedding is still mixed in at every block, even though "
        "the causal mask still only lets it attend to itself."
    )


if __name__ == "__main__":
    main()
