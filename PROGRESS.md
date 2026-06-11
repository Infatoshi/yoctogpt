# YoctoGPT — Session Progress

## Current State (Last Updated: June 2026)

Ready for review before recording. `v1.ipynb` is now the canonical course notebook (52 cells), `v1.py` is the 98-line trophy artifact, and all 7 diagrams render from `.excalidraw` sources via a headless pipeline.

### Positioning (locked in)
- **Notebook = the course.** Readable expanded code, explanations, diagrams, experiments. Teach and record from this.
- **v1.py = the trophy.** Comment-free compressed form, revealed at the end ("everything you learned, on one screen"). Never walk beginners through the golfed code line by line.
- True audience: knows basic Python, knows zero ML. Intro speech order for the video: cold open (terminal demo) -> 30-45s motivation -> roadmap.

---

## June 2026 session: review-driven rework

### Notebook (`v1.ipynb`, 52 cells) — merged from Colab copy + local, then upgraded
- Intro compressed from 6 markdown cells to 2 (hook with before/after, no-math version, big-picture image)
- Terminology dictionary moved to bottom as appendix (define-at-first-use instead of front-loaded glossary)
- "You Be the Model" beat before the loss section (reader predicts `Peter Pi_`)
- ChatGPT bridge paragraph at generate_text (predict-sample-append-repeat IS ChatGPT streaming)
- Training loop now matches v1.py: RMSprop-style optimizer (markdown said RMSprop but code was plain SGD), 1,000 steps, `step % 100 == 0` val checks
- Val loss averaged over 4 windows (was 1 fixed window; bounced non-monotonically and would have generated "is it broken?" comments)
- Loss curve plot cell (train noisy + val + random-guessing line; matplotlib for plotting only)
- NEW section 9: attention heatmap for "Peter Pi" via `gpt_forward(..., return_attention=True)` — clear sparse lower-triangular structure, final `i` attends to `P`
- NEW section 10: play cells + temperature TRY IT (temperature param added to generate_text)
- NEW section 11: train-on-your-own-text instructions
- `txt` synced to v1.py version (newlines + "Where's") so vocab stays 24 and params 3,968, matching the diagrams
- Deleted scratch cell (`a = 3; b = 1; type(a)`); all "500 steps" references fixed to 1,000

### Verified by execution (June 10, 2026, M4 Max)
- Full notebook executes with 0 errors; vocab 24, params 3,968, initial loss 3.34 vs ln(24)=3.18
- Training arc: val 3.23 -> 1.24 (step 100) -> 0.53 (step 400) -> ~0.43 (step 1000); gibberish -> "Peter Piper picked a..."
- v1.py: 200 steps = 10.5s locally, so full 2001-step run ~105s (can train live on camera)

### v1.py (98 lines)
- Val loss averaged over 4 windows (same fix as notebook). Verified: 3.23 -> 0.87 -> 0.78 -> 0.52 monotonic-ish.

### README
- Restructured: hook first, "two ways in" (course vs trophy) framing, course funnel moved to bottom ("Going deeper")

### Diagrams (all 7 re-rendered at 2x from .excalidraw via headless Chrome + excalidraw 0.18)
- `02-chain-rule`: removed stray red lines; backward red arrows now flow along the actual graph edges with grad multipliers at the ops ("grad ×(2·sum) = ×10", "grad ×1 to each")
- `04-attention`: added grayed crossed-out future token ("can't look ahead!") — causality now visible; caption extended
- `06-full-gpt`: parameter box completed (was authored with literal "..."): 384+128+1,024+2,048+384 = 3,968
- `07-training-loop`: NEW — forward/loss/backward/update cycle, "repeat ×1,000" center
- Render pipeline (rebuild any PNG): `/tmp/exca/render.py` pattern — exportToSvg from esm.sh @excalidraw/excalidraw@0.18.0 in headless Chrome, inline SVG, screenshot at computed size. Kroki.io excalidraw endpoint was down (504s).

---

### Terminology depth pass (post-review)
- Section 4 rewritten: matmul walked numerically (new diagram `08-matrix-vector`), "projection" established from dot products, RMS normalization actual math + explode/vanish why, nonlinearity explained via the linear-collapse argument with a runnable proof cell (two matmuls == one matmul)
- New diagram `09-scalars-to-matrices`: Value scalar -> list -> matrix, the micrograd-to-tensor storage bridge
- Term mechanics added: token (arcade token stand-in), embedding (discrete symbols embedded into continuous space), position embedding (attention is order-blind), attention (= learned weighted average, fixed budget of focus)
- Appendix dictionary expanded: projection, RMS normalization, nonlinearity, logits, KV cache entries added

## Pending

- [ ] User review, then: commit + push (the notebook references `images/07-training-loop.png` via raw GitHub URL — 404s until pushed; edited diagrams show stale until pushed)
- [ ] After push: `gog upload v1.ipynb -a infatoshi@gmail.com --replace=133_4cWN8nQJ-4DyAFUaz4dRMOcGccNVY` (updates the Colab file in place, link preserved)
- [ ] Record the video following `planning.md`
- [ ] Record a short "Python idioms" aside (range, zip, enumerate, one-line loops/comprehensions) for beginners; slot it early, before the Value class
- [ ] Before recording: one test run of the notebook in actual Colab to quote its (slower) training time to viewers

---

## Teaching Notes (kept from earlier session)

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
cd /Users/infatoshi/dev/teaching/yoctogpt
uv run v1.py                # runs training (2001 steps, ~105s on M4 Max)
uv run --with jupyter,matplotlib jupyter notebook v1.ipynb
```
