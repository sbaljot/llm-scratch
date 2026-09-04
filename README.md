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

## 2. Core Model Architecture Components

This is the **heart of the neural network**, where the Transformer architecture is implemented using PyTorch modules (`nn.Module`).

### Embedding Layers

The model uses two embedding layers:

* **Token Embedding:** Maps each token ID to a dense vector representation.
* **Positional Embedding:** Encodes the position of each token in the sequence, allowing the model to understand word order.

```text
Token IDs
    ↓
Token Embedding ─────┐
                     ├──→ Combined Representation
Position IDs         │
    ↓                │
Positional Embedding┘
```

### Causal Self-Attention

**Causal Self-Attention** is the core mechanism that allows the Transformer to determine which previous tokens are relevant when predicting the next token.

The input is projected into three matrices:

* **Query (Q):** Represents what the current token is looking for.
* **Key (K):** Represents what information each token contains.
* **Value (V):** Contains the information that is ultimately aggregated.

The attention mechanism uses these matrices to calculate attention scores and applies a **causal mask** to prevent a token from attending to future tokens.

```text
Input
  ↓
 ┌───────────────┐
 │ Q      K      V│
 └───────┬───────┘
         ↓
  Attention Scores
         ↓
    Causal Mask
         ↓
       Softmax
         ↓
   Weighted Values
         ↓
      Output
```

The causal mask ensures that when predicting a token, the model can only use the current and previous tokens, never future tokens.

### Multi-Head Attention & Transformer Blocks

Instead of using a single attention mechanism, **Multi-Head Attention** runs multiple attention heads in parallel. Each head can learn different relationships between tokens.

The attention mechanism is combined with other components to form a **Transformer Block**:

* **Multi-Head Self-Attention**
* **Feed-Forward Network (FFN)**
* **Layer Normalization**
* **Residual Connections**

Multiple Transformer blocks can then be stacked together to increase the model's representational capacity.

```text
Input
  ↓
Layer Normalization
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
Layer Normalization
  ↓
Feed-Forward Network
  ↓
Residual Connection
  ↓
Output
```

### Output Linear Head

After passing through the Transformer blocks, the final hidden representations are passed through a **linear layer**.

This layer maps the model's hidden dimension to the size of the vocabulary, producing a **logit for every possible token**.

```text
Final Hidden States
        ↓
   Linear Layer
        ↓
Vocabulary-sized Logits
        ↓
      Softmax
        ↓
Token Probabilities
```

The token with the highest probability, or a token sampled using a chosen sampling strategy, can then be selected as the model's next token.

## 3. Pretraining & Optimization Components

Once the model architecture is implemented, the next step is to train the model by optimizing its parameters.

### Loss Function

The model uses **Cross-Entropy Loss** to measure how accurately it predicts the actual next token in the training data.

The model produces logits for every token in the vocabulary, and the loss compares these predictions against the correct target token.

```text
Input Tokens
     ↓
   Model
     ↓
Predicted Logits
     ↓
Cross-Entropy Loss
     ↑
Target Tokens
```

A lower loss indicates that the model is becoming better at predicting the next token.

### Optimizer & Learning Rate Scheduler

An optimizer is responsible for updating the model's parameters based on the gradients calculated during backpropagation.

This project uses:

* **AdamW:** Optimizer used to update the model's weights.
* **Cosine Annealing:** Learning rate scheduler that gradually adjusts the learning rate during training.

```text
Loss
 ↓
Backpropagation
 ↓
Gradients
 ↓
AdamW
 ↓
Updated Model Weights
```

### Training Loop

The training loop repeatedly processes batches of training data and updates the model's parameters.

The general process is:

```text
Load Batch
   ↓
Forward Pass
   ↓
Calculate Loss
   ↓
loss.backward()
   ↓
Calculate Gradients
   ↓
Optimizer Step
   ↓
Update Weights
   ↓
Repeat
```

The training process continues for a predefined number of epochs or training steps until the model has sufficiently learned the patterns present in the training corpus.

---

## 4. Inference & Generation Components

After pretraining, the model can be used to generate text through an **autoregressive generation pipeline**.

### Autoregressive Generation Loop

The generation process starts with an initial text prompt. The model predicts the next token, which is then appended to the existing sequence. This process is repeated until the desired number of tokens has been generated.

```text
Initial Prompt
      ↓
   Tokenizer
      ↓
    Token IDs
      ↓
     Model
      ↓
     Logits
      ↓
Sample Next Token
      ↓
Append Token
      ↓
    Model Again
      ↓
     Repeat
```

For example:

```text
Prompt:
"The cat"

        ↓

"The cat sat"

        ↓

"The cat sat on"

        ↓

"The cat sat on the"

        ↓

"The cat sat on the mat"
```

This process is called **autoregressive generation** because each newly generated token becomes part of the input used to predict the next token.

### Sampling Strategies

Instead of always selecting the token with the highest probability, different sampling strategies can be used to control the quality and randomness of generated text.

#### Temperature

**Temperature** controls the randomness of token selection.

* Lower temperature → more deterministic and predictable output.
* Higher temperature → more random and diverse output.

#### Top-k Sampling

**Top-k sampling** restricts the possible next tokens to the `k` tokens with the highest probabilities.

```text
Vocabulary
    ↓
Select Top-k Tokens
    ↓
Sample from Top-k
    ↓
Next Token
```

#### Top-p Sampling

**Top-p sampling** selects the smallest group of tokens whose cumulative probability reaches a specified threshold `p`.

This dynamically changes the number of candidate tokens based on the model's probability distribution.

```text
Vocabulary
    ↓
Sort by Probability
    ↓
Calculate Cumulative Probability
    ↓
Keep Tokens up to p
    ↓
Sample Next Token
```

These sampling techniques can be combined with temperature to control the balance between **coherence, predictability, and creativity** during generation.
