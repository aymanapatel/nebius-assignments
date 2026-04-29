# Insights

- A decoder-only Transformer is mostly about preserving tensor shape `(B, T, C)` while repeatedly applying two sublayers: causal self-attention and position-wise MLP.
- Causal masking must be applied **before** softmax on attention scores; this is what prevents information leakage from future tokens.
- Multi-head attention works by splitting channels into heads, attending independently per head, then merging back to the residual stream dimension.
- Pre-norm residual blocks (`x = x + sublayer(LN(x))`) are more stable to train than post-norm for modern GPT-style stacks.
- Positional embeddings are required because self-attention alone is permutation-invariant over sequence positions.
- In autoregressive generation, context must be cropped to the last `block_size` tokens each step so positional indices stay in range.
- For an untrained model, cross-entropy should start near `log(vocab_size)` (uniform prediction baseline); this is a strong sanity check for forward/loss correctness.
- Perplexity and bits-per-character translate loss into interpretable quality metrics:
  - `PPL = exp(NLL)`
  - `bpc = NLL / ln(2)`
