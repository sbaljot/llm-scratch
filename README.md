# llm-scratch

## 1. Text Processing & Data Components

Before any mathematical operations take place, the raw text must first be converted into numerical inputs that can be processed by the neural network.

### Tokenizer

A tokenizer converts raw text into smaller units called **tokens** and assigns each token a unique integer ID from a vocabulary.

For this project, the tokenizer can use a method such as **Byte-Pair Encoding (BPE)** to split text into sub-word tokens.

```text
Raw Text
   ↓
Tokenizer
   ↓
Tokens
   ↓
Token IDs
```

### Dataset & DataLoader

The dataset pipeline prepares the tokenized text for **next-token prediction**.

The training corpus is divided into fixed-length **context windows**, where:

* The **input** contains a sequence of tokens.
* The **target** contains the same sequence shifted by one token.

For example:

```text
Input:  "The cat sat"
Target: "cat sat on"
```

The `Dataset` is responsible for creating these input-target pairs, while PyTorch's `DataLoader` is used to batch and efficiently provide them to the model during training.
