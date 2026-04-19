# YoctoGPT — Session Progress

## Current State (Last Updated: April 2026)

The project is ready for video recording. Both `v1.py` and `v1.ipynb` are complete and synchronized.

---

## Completed Work

### 1. Code Cleanup — Variable Naming
Both files now use **matching, readable variable names**:

| Variable | Purpose |
|----------|---------|
| `Value` | Autograd scalar class |
| `characters`, `vocab_size` | Tokenizer vocabulary |
| `context_length`, `embed_dim`, `num_layers`, `learning_rate` | Hyperparameters |
| `train_data`, `val_data` | Data splits |
| `make_matrix` | Weight initializer |
| `weights` | Model state dict |
| `token_embed`, `position_embed`, `output_proj` | Embedding/output weight keys |
| `layer{i}.query/key/value/attn_out/ff_up/ff_down` | Per-layer weight keys |
| `linear`, `normalize`, `softmax` | Neural net primitives |
| `gpt_forward` | Forward pass |
| `key_cache`, `value_cache` | KV cache for attention |
| `attention_scores`, `attention_weights` | Attention intermediates |
| `compute_loss` | Loss function |
| `generate_text` | Inference/generation |
| `all_parameters` | Flat param list for optimizer |

### 2. Notebook (`v1.ipynb`)
- **42 cells** with extensive markdown explanations
- Beginner-friendly: assumes only Python knowledge
- Progressive complexity: builds from tokenization → autograd → full GPT
- 6 embedded Excalidraw diagrams in `images/`:
  - `01-computation-graph.png` — basic autograd visualization
  - `02-chain-rule.png` — chain rule example
  - `03-token-embedding.png` — embedding lookup illustration  
  - `04-attention.png` — attention mechanism
  - `05-gpt-block.png` — transformer block with residuals
  - `06-full-gpt.png` — complete model overview
- Interactive "TRY IT YOURSELF" cells for experimentation
- Terminology dictionary at the top

### 3. Script (`v1.py`)
- **97 lines** (under 100 target)
- `# fmt: off` at top to prevent auto-formatter expansion
- Same logic as notebook, just compact
- Runs 2001 training steps (user changed from 501)

### 4. Residual Connections
Confirmed present in both files:
```python
residual, x = x, normalize(x)  # save pre-norm
# ... attention ...
x = [...attention_output + residual...]  # skip connection 1
# ... feedforward ...
x = [...ff_output + x...]  # skip connection 2
```
This is Pre-LN (normalize before attention/FFN, not after).

---

## Files Structure

```
yoctogpt/
├── v1.py              # 97-line polished script
├── v1.ipynb           # Educational notebook with diagrams
├── README.md          # Project overview
├── planning.md        # Video teaching plan (8 sections)
├── PROGRESS.md        # This file
└── images/
    ├── 01-computation-graph.png
    ├── 01-computation-graph.excalidraw
    ├── 02-chain-rule.png
    ├── 02-chain-rule.excalidraw
    ├── 03-token-embedding.png
    ├── 03-token-embedding.excalidraw
    ├── 04-attention.png
    ├── 04-attention.excalidraw
    ├── 05-gpt-block.png
    ├── 05-gpt-block.excalidraw
    ├── 06-full-gpt.png
    └── 06-full-gpt.excalidraw
```

---

## Pending / Ideas

- [ ] Record the video following `planning.md` structure
- [ ] The notebook text slightly differs from v1.py's txt (notebook has periods/commas in different places) — may want to sync if it matters
- [ ] Consider adding more "TRY IT YOURSELF" experiments
- [ ] Could add a cell showing the learned embeddings (PCA/t-SNE of token_embed)

---

## Teaching Notes (from last session)

### Explaining `backward()` to students:

**Part 1 — Why topological sort?**
> "You can't compute the gradient for `a` until you've computed the gradient for the nodes that depend on `a`. Topological sort gives us the right order: process parents before children."

**Part 2 — The code:**
```python
def build_order(node):
    if node not in visited:
        visited.add(node)
        for child in node.children:
            build_order(child)      # visit children first
        topo_order.append(node)     # THEN add yourself
```
> "This is depth-first search. 'Before I add myself, make sure all my children are there.' When reversed, parents come before children."

**Part 3 — Gradient flow:**
```python
for node in reversed(topo_order):
    for child, local_grad in zip(node.children, node.local_grads):
        child.grad += local_grad * node.grad
```
> "`local_grad` = 'how does this node change when child changes?' (computed during forward)"
> "`node.grad` = 'how does output change when this node changes?' (computed during backward)"
> "Multiply them = chain rule. The `+=` handles the case where a value is used multiple times."

**One-liner summary:**
> "Walk to all the leaves first, then backtrack. As you backtrack, multiply and accumulate: each node tells its children how much blame they share for the final output."

---

## Quick Resume Commands

```bash
cd /Users/infatoshi/yoctogpt
python v1.py  # runs training (2001 steps)
```

To test notebook:
```bash
jupyter notebook v1.ipynb
```
