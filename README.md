# Generatively Pretrained Transformer (GPT) — From Scratch

A ground-up implementation of a GPT-style transformer language model, progressing from a simple bigram model to a full transformer with custom CUDA kernels.

## Motivation

Understanding transformer internals by building every component from scratch — no `nn.TransformerEncoder`, no shortcuts. This includes eventually writing custom CUDA kernels for core operations.

## Current Progress

### Phase 1: Bigram Language Model ✅

Character-level language model trained on War and Peace (3.2M characters, 111 unique tokens).

| Metric | Value |
|--------|-------|
| Vocabulary | 111 characters |
| Training data | 2.9M characters |
| Validation data | 323K characters |
| Initial loss | 5.33 (random) |
| Final loss | 2.39 (after 100K steps) |

**Sample output (bigram only):**
```
"We bothf
mpof uprerust tllowhilyan m teleld ollkn.
fe verng he wad. ame stheve The se wim Pin...
```

Word-like patterns emerge, but no coherent meaning — the model only sees one character of context.

### Phase 2: Self-Attention 🔄

- [ ] Scaled dot-product attention
- [ ] Causal masking (decoder-style)
- [ ] Single attention head

### Phase 3: Multi-Head Attention

- [ ] Multiple parallel attention heads
- [ ] Head concatenation + projection

### Phase 4: Transformer Block

- [ ] LayerNorm
- [ ] Feedforward network (MLP)
- [ ] Residual connections
- [ ] Dropout

### Phase 5: Full GPT

- [ ] Positional embeddings
- [ ] Stacked transformer blocks
- [ ] Scaled initialization

### Phase 6: CUDA Kernels 🔥

- [ ] Custom attention kernel
- [ ] Fused softmax
- [ ] Flash attention implementation
- [ ] Memory-efficient backprop

## Architecture (Target)

```
Input tokens
     │
     ▼
┌─────────────────┐
│ Token Embedding │
│ + Position Emb  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Transformer    │ ×N blocks
│     Block       │
│ ┌─────────────┐ │
│ │ Multi-Head  │ │
│ │ Attention   │ │
│ └──────┬──────┘ │
│        │        │
│ ┌──────▼──────┐ │
│ │ Feedforward │ │
│ └─────────────┘ │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LayerNorm     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linear → Logits │
└─────────────────┘
```

## File Structure

```
generatively_pretrained_transformer/
├── src/
│   └── training_lab.ipynb    # Main development notebook
├── cuda/                      # Custom CUDA kernels (coming)
│   ├── attention.cu
│   └── softmax.cu
├── war_and_peace.txt         # Training corpus
├── AGENTS.md                 # Multi-agent coordination
└── README.md
```

## Setup

```bash
git clone https://github.com/LewallenAE/generatively_pretrained_transformer.git
cd generatively_pretrained_transformer
python -m venv venv
source venv/bin/activate
pip install torch
```

## Training

```python
# In training_lab.ipynb

# Hyperparameters (current)
batch_size = 32
block_size = 8
learning_rate = 1e-3
max_iters = 100000

# Training loop
for steps in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al.

## License

MIT
