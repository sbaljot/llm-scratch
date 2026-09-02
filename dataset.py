"""
dataset.py
PyTorch Dataset + DataLoader that turn a raw text corpus into (input, target)
pairs for next-token-prediction training.

Works with ANY tokenizer object exposing .encode(text) -> list[int],
so both tiktoken and the custom BPETokenizer in tokenizer.py work here.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    """
    Slices one long token sequence into overlapping (input, target) windows.

    Given token ids [t0, t1, t2, t3, t4, t5, ...], max_length=4, stride=1:
      input  = [t0, t1, t2, t3]   target = [t1, t2, t3, t4]
      input  = [t1, t2, t3, t4]   target = [t2, t3, t4, t5]
      ...
    Each target is the input shifted one position to the right -- that shift
    is the entire "next-token prediction" objective.

    stride controls overlap: stride == max_length means no overlap between
    windows; stride < max_length means windows overlap (more training
    examples from the same corpus, at the cost of some redundancy).
    """

    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        assert len(token_ids) > max_length, (
            f"Corpus produced only {len(token_ids)} tokens, need > {max_length}. "
            "Use more text or a smaller max_length."
        )

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
):
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,   # avoids a ragged final batch during training
        num_workers=num_workers,
    )