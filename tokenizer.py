"""
tokenizer.py
A minimal byte-level Byte-Pair Encoding (BPE) tokenizer, built from scratch.

This is the same core algorithm GPT-2 / GPT-4 tokenizers use, simplified for
learning. It operates on raw UTF-8 bytes, so a base vocabulary of just 256
tokens (one per byte value) can already represent ANY text — English,
emoji, code, other languages, etc. Training then learns to merge frequently
co-occurring byte pairs into single tokens, which is what makes encoding
efficient (fewer tokens per word).

Usage:
    tok = BPETokenizer()
    tok.train(corpus_text, vocab_size=1000)
    ids = tok.encode("hello world")
    text = tok.decode(ids)
"""

from collections import Counter


class BPETokenizer:
    def __init__(self):
        # merges: maps a pair of token ids -> the new token id it merges into
        self.merges = {}  # {(id1, id2): new_id}
        # vocab: maps a token id -> the raw bytes it represents
        self.vocab = {i: bytes([i]) for i in range(256)}

    # ---------- training ----------

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256, "vocab_size must be >= 256 (base byte vocab)"
        num_merges = vocab_size - 256

        # start as a flat list of byte values (0-255)
        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            pair_counts = self._get_pair_counts(ids)
            if not pair_counts:
                break  # no more pairs left to merge

            # pick the most frequent adjacent pair in the corpus
            top_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i

            ids = self._merge(ids, top_pair, new_id)

            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {top_pair} -> {new_id} "
                    f"({self.vocab[new_id]}) had {pair_counts[top_pair]} occurrences"
                )

    @staticmethod
    def _get_pair_counts(ids):
        counts = Counter()
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
        return counts

    @staticmethod
    def _merge(ids, pair, new_id):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    # ---------- encode / decode ----------

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair_counts = self._get_pair_counts(ids)
            # among pairs present, apply whichever was learned EARLIEST in training
            candidate = min(
                pair_counts,
                key=lambda p: self.merges.get(p, float("inf")),
                default=None,
            )
            if candidate is None or candidate not in self.merges:
                break
            ids = self._merge(ids, candidate, self.merges[candidate])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def __len__(self):
        return len(self.vocab)