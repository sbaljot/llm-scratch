"""
hf_data.py
Load a Hugging Face dataset and turn it into one long text string that
dataset.py's GPTDataset / create_dataloader can consume.

Install:
    pip install datasets

If the dataset is gated (like Anthropic/enabling-independent-research),
authenticate first:
    huggingface-cli login
"""

from datasets import load_dataset


def load_hf_text(
    path: str,
    name: str | None = None,
    split: str = "train",
    text_columns: list[str] | None = None,
    max_examples: int | None = None,
    join_str: str = "\n\n",
) -> str:
    """
    path:         HF dataset repo id, e.g. "wikitext" or "Anthropic/enabling-independent-research"
    name:         config/subset name, e.g. "wikitext-103-raw-v1" or "metr"
    split:        which split to load (most text-corpus datasets only have "train")
    text_columns: which column(s) hold the text. If None, auto-detects the
                  first string column found in the first row.
    max_examples: cap the number of rows pulled (useful for a quick first run)
    join_str:     separator inserted between concatenated examples

    Returns a single string -- the full corpus text, ready for tokenizer.encode().
    """
    ds = load_dataset(path, name, split=split)

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    if text_columns is None:
        # auto-detect: first column in the first row whose value is a string
        first_row = ds[0]
        text_columns = [k for k, v in first_row.items() if isinstance(v, str)]
        if not text_columns:
            raise ValueError(
                f"No string columns found in {path}/{name}. "
                f"Available columns: {list(first_row.keys())}. "
                "This dataset may be tabular/numeric rather than free text -- "
                "pass text_columns explicitly if you know which field to use."
            )
        print(f"Auto-detected text column(s): {text_columns}")

    pieces = []
    for row in ds:
        for col in text_columns:
            val = row.get(col)
            if val:
                pieces.append(val)

    corpus = join_str.join(pieces)
    print(f"Loaded {len(pieces)} text pieces, {len(corpus):,} characters total.")
    return corpus


if __name__ == "__main__":
    # Example: a real pretraining-style corpus (small subset for a quick test)
    text = load_hf_text(
        "wikitext",
        name="wikitext-103-raw-v1",
        split="train",
        text_columns=["text"],
        max_examples=2000,
    )
    print(text[:500])