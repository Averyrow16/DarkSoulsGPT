# DarkSoulsGPT

A character-level GPT trained on Dark Souls NPC dialogue, built without high-level abstraction in PyTorch.

---

## Summary

A transformer language model built from scratch while walking through Andrej Karpathy's makemore and GPT series. My goal was to understand every component of the architecture end-to-end.

Trained to generate text that sounds like NPC dialogue from Dark Souls.

---

## Architecture

The model follows the GPT architecture from the research paper Attention Is All You Need (Vaswani et al., 2017), but scaled down to run on a single GPU.

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
Each head computes queries, keys, and values from the input. The attention scores
`q @ k.T / sqrt(head_size)` determine how much each token attends to every previous token. The torch.tril mask ensures tokens never see future tokens during training.

**Multi-Head Attention**
Six attention heads run in parallel, each learning different patterns. Their outputs are concatenated and projected back to n_embd.

**Feed-Forward Network**
After attention (the communication between tokens), each token passes independently through a small MLP (computation on what was communicated). The hidden layer expands to 4 x n_embd before projecting back.

**Residual Connections**
Each sub-layer adds its input back to its output: `x = x + sublayer(x)`. This gives gradients a direct path back through the network, which helps to avoid vanishing gradients or info loss when training 6+ layers.

**LayerNorm**
Applied before each sub-layer (pre-normalization). Normalizes activations across the embedding dimension, which stabilizes training significantly.

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
- **Size:** ~61,500 words
- **Vocabulary:** 67 unique characters
- **Split:** 90% train / 10% validation

---

## Output From My Best Loss (1.48)
```
Sorcery... You are welcome... Oh, the way... They cursed.
Keh heh...
Keh heh heh...It's taken to be it.
Now the twicenail is sad?
Keh heh heh...
Oh, you again? Damn.
We're this fool...
My bloody are except, master...
Spurized these days...
Daty business of sun...human...
Yah, young Undead With the curse purchase to remeld the work upon you...
Oh you again?
Very well. Just call take you give up yourself to this.
Quelaag? It is your sword.
Don't you seek to this for me.
Well, you desired to the Ca
```
---

## References

- Attention Is All You Need — Vaswani et al.: https://arxiv.org/abs/1706.03762
- Let's build GPT from scratch — Karpathy: https://www.youtube.com/watch?v=kCc8FmEb1nY
- nanoGPT: https://github.com/karpathy/nanoGPT
