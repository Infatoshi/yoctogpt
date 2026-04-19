# yoctogpt — teaching plan

## positioning

this is the "level 0" entry point. nanogpt assumes pytorch fluency. fcc course assumes willingness to grind through 7+ hours and pytorch/numpy basics. yoctogpt assumes only python — nothing else. the whole model is 100 lines, standard library only. the point is to make GPT pre-training *obvious* before it's practical.

the hook: "most GPT codebases are practical before they are obvious. this one goes the other direction."

the payoff: if you deeply understand these 100 lines, you have a real mental model for how every LLM works. the fcc course and nanogpt become much easier after this.

## passives (production / presentation)
- zoom in on code for phone users (big text, no tiny IDE font)
- dark background, high contrast syntax highlighting
- verbally reward progress between sections ("if you're still here, you now understand backprop at a scalar level — that's huge")
- keep terminal output visible when training runs — the model visibly improving is the dopamine hit
- show the gibberish-to-coherent transition early in the intro as the hook, then rewind to "how did we get here"
- no scrolling walls of code — show one function at a time, let it breathe
- speak to the code on screen, not to slides — this is a "code along" not a lecture
- pinned comment or description with links to: v1.py, v1.ipynb, README, fcc course, nanogpt, backprop video
- consider picture-in-picture webcam in corner (not required but adds personality)
- use chapter markers / timestamps in the video description matching the 8 sections

## intro (~2 min)

1. **cold open on the result**: run `python3 v1.py`, show the terminal printing gibberish at step 0, then "Peter Piper picked" emerging by step 400. let it land. "this is a GPT learning a tongue twister from pure randomness. the entire thing is 100 lines of python. no pytorch, no numpy, no pip install anything."

2. **what this is and isn't**: "this is not a production model. it's a teaching tool. if you understand these 100 lines deeply, you understand the architecture behind ChatGPT, Claude, Gemini — all of them. the scale is different but the ideas are identical."

3. **prerequisites**: "you need to know python. that's it. if you know what a list, a for loop, and a class are, you can follow this. i don't assume any math beyond arithmetic — we build everything we need as we go."

4. **roadmap**: flash the 8 sections briefly. "we'll start with the data, build our own autograd engine, set up the model weights, write the building blocks, assemble the GPT forward pass, define the loss, write inference, and then train it. 8 pieces. by the end you'll understand all of them."

5. **plug the ecosystem**: "if after this you want to go deeper with pytorch and real-scale data, i made a full course on freeCodeCamp (link). if you want to understand how to make all this fast on GPUs, i have a CUDA course too. but you don't need any of that for today."

## teaching flow (section by section)

### 1 · data & tokenization
- show the raw text (peter piper tongue twister)
- explain character-level tokenizer: "each unique character gets a number. that's it. production LLMs use subword tokenizers (BPE / sentencepiece) but the concept is identical — text goes in, integers come out"
- show the encoded integers, point out the mapping is reversible
- train/val split: "we hide 10% of the data. if the model can predict those unseen characters well, it actually learned the *pattern*, not just memorized"
- set hyperparameters: ctx=8, d=16, nl=1, lr=0.01. "these are intentionally tiny. 8 characters of context, 16-dimensional embeddings, 1 layer. the whole model is ~4000 parameters. GPT-4 has maybe a trillion. same architecture though."

### 2 · autograd engine (class V)
- this is the hardest section conceptually. go slow.
- start with the question: "how does a neural net learn? it needs to know, for every single number in the model, *which direction should I nudge this to make the output better*. that's what gradients are."
- show the V class wrapping a scalar. "every number in our model is one of these V objects. it tracks its value and its gradient."
- walk through __add__ and __mul__ — "when we add two V's, the result remembers its parents and how to pass gradients back"
- demo: `x = V(3.0); y = x**2; y.backward(); print(x.grad)` → 6.0. "derivative of x^2 is 2x. at x=3, that's 6. our autograd got it right."
- "this is a tiny version of what pytorch does under the hood. pytorch is faster (tensors, GPU, C++), but the idea is exactly this."
- link to the backprop whiteboard video for anyone who wants more depth

### 3 · model parameters
- "a neural net is just a pile of numbers (weights) arranged in matrices. we initialize them as small random values."
- show `mat(r, c)` — it's just a 2D list of V objects
- walk through the state dict: "wte is the token embedding table — every character gets a 16-dim vector. wpe is position embeddings — every position (0 through 7) also gets a vector. these two get added together so the model knows *what* token it's looking at and *where* it sits."
- per-layer weights: q, k, v, o (attention), f1/f2 (feedforward). name-drop "query, key, value" but say "we'll see what these actually do in section 5"
- "total parameters: ~4000. that's literally less than the number of words in a short essay."

### 4 · neural net primitives (lin, norm, soft)
- `lin`: "matrix-vector multiply. this is what the 'linear layer' or 'nn.Linear' does in pytorch. it's just dot products."
- `norm`: "RMS normalization. keeps the numbers from blowing up or shrinking to zero during training. production models use LayerNorm — same idea."
- `soft`: "softmax. turns a list of raw scores into probabilities that sum to 1. we subtract the max first for numerical stability (so exp() doesn't overflow)."
- quick demo: softmax([1,2,3]) sums to 1.0. "the bigger number gets a bigger probability. that's how the model picks which token is most likely."

### 5 · GPT forward pass
- this is the core. spend time here.
- "the gpt function takes one token, its position, and the KV cache. it outputs scores (logits) for every possible next character."
- step through: embed + position → norm → for each layer: Q/K/V projections → attention scores (Q dot K / sqrt(d)) → softmax → weighted sum of V → output projection + residual → feedforward (expand 4x, relu, project back) + residual → lm_head
- key analogy (from your fcc notes): "query = what am I looking for? key = what do I contain? value = what will I share if you find me interesting? the dot product between Q and K measures how 'interesting' two tokens are to each other."
- emphasize: "this is causal attention — each token can only look at tokens before it, never ahead. that's why it's called autoregressive."
- "the residual connections (the `a + b` lines) are crucial — they let gradients flow straight through without vanishing. deep nets don't train without them."

### 6 · loss function
- "cross-entropy loss. for each position, we ask: what probability did the model assign to the *correct* next character? we take -log of that. lower is better."
- show that a random model should have loss ≈ ln(vocab) ≈ 3.18. "1 in 24 chance of guessing right."
- "the loss function is what backward() will differentiate through. it connects the model's predictions to the training signal."

### 7 · inference / generation
- "to generate text, we feed the prompt in token by token (building the KV cache), then sample from the softmax distribution."
- show pre-training generation: gibberish. "the model's weights are random — it has no idea what English looks like yet."
- "after training, the same function will produce recognizable fragments of the tongue twister."

### 8 · training loop
- "the training loop is dead simple: pick a random window of training text, compute the loss, call backward() to get gradients, update every parameter."
- explain the optimizer: "this is a simplified RMSProp. it keeps a running average of squared gradients so parameters that get big gradients take smaller steps. prevents oscillation."
- "every 100 steps we check val loss and generate a sample — this is how you know training is working."
- let it run. watch the loss drop. watch the generation go from noise → "Peter Piper" → recognizable phrases.
- closing: "you just trained a transformer language model from scratch. 100 lines. no libraries. everything you saw here — the autograd, the attention, the feedforward, the training loop — this is what's inside GPT-4, Claude, Gemini. scaled up enormously, but the same fundamental ideas."

## what to point out (karpathy-style "aha" moments)
- the model starts as pure random numbers and *learns* english character patterns just from gradient descent
- attention is just a weighted average where the weights are data-dependent (Q dot K)
- residual connections are what make deep networks trainable
- the KV cache lets us avoid recomputing attention for previous tokens during inference
- the loss goes from ~3.2 (random) toward ~1.0 — meaning the model goes from "1 in 24 chance" to "roughly 1 in 3 chance" of predicting the next char correctly. on a 200-char dataset that's surprisingly good
- "softmax + sampling" is how you get probabilistic generation — same prompt, different completions each time

## what NOT to cover (keep it simple)
- multi-head attention (we use a single head — mention MHA exists, link to fcc course)
- batching (we process one sequence at a time — mention batching is how you scale)
- dropout / weight decay / learning rate schedules
- BPE / subword tokenization (mention it, link to nanogpt)
- GPU acceleration / CUDA (mention it, link to CUDA course)
- MoE / GQA / RoPE / any modern arch twists (save for a future "what changed since" video)

## follow-up funnel
1. yoctogpt (this) — "understand the ideas"
2. fcc LLM course — "build it for real with pytorch and bigger data"
3. nanogpt — "production-adjacent training code"
4. CUDA course — "make it fast"