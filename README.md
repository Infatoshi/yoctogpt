# yoctogpt

`v1.py` is a one-file, pure-Python, deliberately tiny GPT training script.

It is basically `micrograd` plus a very small `nanoGPT`-style decoder, reduced until the whole thing fits in one obvious file:

- scalar autograd
- character-level tokenization
- token and position embeddings
- causal self-attention
- MLP
- next-token loss
- RMSProp-like parameter update
- streaming text generation

The point is not speed. The point is to make the whole training loop small enough that a student can read it end to end and still recognize every moving part.

## Why This Exists

Most GPT codebases are practical before they are obvious.

This one goes the other direction:

- one file
- standard library only
- hardcoded dataset
- hardcoded hyperparameters
- no config system
- no checkpoints
- no batching machinery
- no optimizer classes
- no framework abstractions
- no comments inside the code

If a decision made the file easier to read and mentally simulate, it stayed. If it made the file more practical but harder to hold in your head, it got cut.

## Design Decisions

### 1. Pure Python Lists

There is no `torch`, no `numpy`, and no custom extension code.

Matrices are lists of lists. Parameters are scalar objects. Forward and backward passes happen in Python loops. This is slow on purpose. The slowness is educational because it makes the cost of each abstraction visible.

### 2. Scalar Autograd Instead of Tensor Autograd

The `V` class is the whole differentiation engine. Every scalar value knows:

- its numeric value
- its gradient
- the children that produced it
- the local derivatives needed for backprop

The math is spelled out through a tiny scalar object with overloaded arithmetic plus a few explicit activations like `log`, `exp`, and `relu`. That keeps the lecture centered on autograd and neural nets instead of on framework internals. It also keeps the code brutally slow, which is acceptable here.

### 3. Tiny Overfitting Dataset

The dataset is a short Peter Piper rhyme embedded directly in the script. That choice is intentional:

- the full corpus is visible in one glance
- the vocabulary is tiny
- the model can overfit in a short run
- generation becomes interpretable quickly

This is not a general language model. It is a learning artifact.

### 4. Character-Level Tokens

Each unique character becomes one token id. Tokenization is just `chars.index(c)`.

That is inefficient, but it removes one more layer of hidden machinery. Students can understand the vocabulary immediately.

### 5. `ctx = 8`

The context window is intentionally small. The model only sees eight characters at a time during training and generation. That keeps the compute cheap enough for pure Python and makes the limitation easy to feel in the output.

### 6. `nh = 1`

There is only one attention head.

Multi-head attention is useful in real models, but it adds slicing logic without changing the core idea of attention. Using one head keeps the attention path easier to read:

- one query stream
- one key stream
- one value stream
- one attention output

That is a better trade for this file.

### 7. Top-Level Training Script

There is no application structure. The file initializes parameters and trains immediately when run.

That is less reusable than a library layout, but it keeps the whole artifact linear:

1. define the math
2. define the model
3. initialize weights
4. train
5. sample

For this project, that is the right shape.

### 8. Streaming Inference

The sampled text prints one character at a time.

This is not technically necessary, but it better matches how people expect model output to appear and makes the training checkpoints feel alive.

## What The Script Prints

Every 100 steps, and on the final step, the script prints:

- validation loss
- a 50-character continuation starting from `Peter Piper `

The generated text streams as it is sampled.

## Run

Use `uv`:

```bash
uv run python v1.py
```

## What To Look For

At the beginning, the output is mostly garbage.

As training continues, the model starts to recover:

- repeated letter patterns
- the phrase `Peter Piper`
- fragments of `pickled peppers`
- the rhyme’s line rhythm

Because the model is tiny and the context is only eight characters, it never becomes clean or robust. That limitation is part of the lesson.

## How To Read The File

Read it in this order:

1. dataset and hyperparameters
2. `V`
3. `mat`, `lin`, `norm`, `soft`
4. `gpt`
5. `loss`
6. `inference`
7. the training loop at the bottom

If you understand those seven pieces, you understand the whole project.

## What This Is Not

This is not:

- fast
- scalable
- numerically polished
- reusable infrastructure
- a replacement for `nanoGPT`
- a replacement for `micrograd`

It is a compressed educational artifact.

## Current Defaults

- dataset: Peter Piper rhyme
- vocab size: derived from the embedded text
- context window: 8
- embedding width: 16
- layers: 1
- attention path: single-head
- training steps: 500
- generation length: 50 characters

## Why The Code Has No Comments

The file is intentionally written so the structure has to carry the meaning. The README is where the explanation lives. That keeps the code itself dense and direct while still preserving the reasoning behind the choices.
