# DarkSoulsGPT

A character-level GPT trained on Dark Souls NPC dialogue, built from scratch in PyTorch.

---

## Summary

A transformer language model built from scratch while walking through Andrej Karpathy's makemore and GPT series. The goal was to understand every component of the architecture end-to-end rather than using a library that abstracts it away.

The model is trained on NPC dialogue from Dark Souls and generates new text in the same style.

---

## Architecture

The model follows the GPT architecture from Attention Is All You Need (Vaswani et al., 2017), scaled down to run on a single GPU.

    Input tokens
        |
        +-- Token Embedding    (vocab_size -> n_embd)
        +-- Position Embedding (block_size -> n_embd)
        |
        x = tok_emb + pos_emb
                |
        +-------+--------+
        |  Block x 6     |   <- repeated n_layer times
        |                |
        |  LayerNorm     |
        |  MultiHead     |   <- 6 attention heads in parallel
        |  Attention     |
        |  + residual    |
        |                |
        |  LayerNorm     |
        |  FeedForward   |   <- Linear -> ReLU -> Linear
        |  + residual    |
        +-------+--------+
                |
        LayerNorm (final)
                |
        Linear head -> logits over vocab
                |
        Softmax -> probability over next token

### Key components

**Self-Attention Head**
Each head computes queries, keys, and values from the input. Attention scores
`q @ k.T / sqrt(head_size)` determine how much each token attends to every previous token. A causal mask (torch.tril) ensures tokens never see future tokens during training.

**Multi-Head Attention**
Six attention heads run in parallel, each learning different patterns. Their outputs are concatenated and projected back to n_embd.

**Feed-Forward Network**
After attention (communication between tokens), each token passes independently through a small MLP (computation on what was communicated). The hidden layer expands to 4 x n_embd before projecting back.

**Residual Connections**
Each sub-layer adds its input back to its output: `x = x + sublayer(x)`. This gives gradients a direct path back through the network, critical for training 6+ layers without vanishing gradients.

**LayerNorm**
Applied before each sub-layer (pre-normalization). Normalizes activations across the embedding dimension, stabilizing training significantly.

### Hyperparameters

| Parameter     | Value |
|---------------|-------|
| n_embd        | 384   |
| n_head        | 6     |
| n_layer       | 6     |
| block_size    | 256   |
| batch_size    | 64    |
| dropout       | 0.2   |
| learning_rate | 1e-3  |
| optimizer     | AdamW |

---

## Dataset

- **Source:** Dark Souls NPC dialogue
- **Size:** ~61,500 words, character-level tokenization
- **Vocabulary:** 67 unique characters
- **Split:** 90% train / 10% validation

Character-level tokenization means the model constructs words letter by letter.

---

## References

- Attention Is All You Need — Vaswani et al.: https://arxiv.org/abs/1706.03762
- Let's build GPT from scratch — Karpathy: https://www.youtube.com/watch?v=kCc8FmEb1nY
- nanoGPT: https://github.com/karpathy/nanoGPT
