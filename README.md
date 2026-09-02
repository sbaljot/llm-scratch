# llm-scratch

1. Text Processing & Data Components
Before any math happens, you need infrastructure to convert raw text into numerical inputs that a neural network can process.

The Tokenizer: A component (like a Byte-Pair Encoding or BPE tokenizer) that splits raw text into sub-words and assigns each a unique integer ID from a vocabulary list.

Dataset & DataLoader: PyTorch dataset classes that chunk your training text corpus into fixed-length context windows and generate input-target sequences for next-token prediction.
