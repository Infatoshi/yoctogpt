# yoctogpt
> Generatively Pre-train a Transformer in 100 lines of python

Most GPT codebases are practical before they are obvious. This one goes the other direction: one file, standard library only, hardcoded dataset (yes, our goal is to overfit -- educational reasons), hardcoded hparams, no checkpoints, no batching, no torch/tensorflow/numpy abstractions.

`v1.py` is [micrograd](https://github.com/karpathy/micrograd) + [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (inspo from [nanogpt](https://github.com/karpathy/nanoGPT)). There are no dependencies. Simply run `python3 v1.py` or `uv run v1.py` to see a GPT learn a tongue twister from zero knowledge about the world (brain starts as random numbers).

## Two ways in

**The course:** the [notebook](https://colab.research.google.com/drive/133_4cWN8nQJ-4DyAFUaz4dRMOcGccNVY?usp=sharing) (`v1.ipynb`). Same model, expanded into readable code with explanations, diagrams, and experiments. If you know basic Python (lists, loops, classes) and nothing about ML, start here. No installs, just press run.

**The trophy:** `v1.py`. The whole thing compressed to under 100 lines, comment-free. Read it *after* the notebook, as a victory lap: everything you learned, on one screen. If you put in the hours and study each line, you will have a solid mental model for how current LLMs and GPTs work.

I admire karpathy's teaching approach where he introduces enough such that the curiosity of the student can fill in the rest. He makes it very fun to learn (I started with nanogpt :D). That's the goal here too.

## What got simplified (and why)

To keep the core ideas bare: a character-level tokenizer, a single causal attention head rather than Multi-head Attention, a basic relu MLP (not MoE), a compact RMSprop-like optimizer to help the nn learn fast without scaring you with complexity, val loss with text streaming to make it "feel" more intuitive, a simple version of position embeddings, and scalar-level autograd to introduce backpropagation (how neural nets learn) in the cleanest way. You get some core lines of code to understand very deeply, and nothing else.

## What to look for

At the beginning, the output is mostly garbage.

As training continues, the model starts to recover:

- repeated letter patterns
- the phrase `Peter Piper`
- fragments of `pickled peppers`
- the rhyme's line rhythm

Because the model is tiny and the context is only eight characters, it never becomes clean or robust. That limitation is part of the lesson.

## How to read v1.py

Read it in this order:

1. dataset and hyperparameters
2. `Value` (the autograd engine)
3. `linear`, `normalize`, `softmax`
4. `gpt_forward`
5. `compute_loss`
6. `generate_text`
7. the training loop at the bottom

If you understand those seven pieces, you understand the whole project.

## Going deeper

- **Backprop on a whiteboard:** I have a whiteboard explanation of backprop for the simplest neural net (an MLP) [here](https://youtu.be/0dbihoMRuyg?si=CvXDG6BC8khGcxoT). I made it to teach myself how backprop works. I knew I understood it fully when I could explain it to an audience on a whiteboard.
- **Real scale with PyTorch:** my [LLMs from scratch course](https://youtu.be/UU1WVnMk4E8?si=RTKgM3YJNJSwsHz3) (freeCodeCamp). I built it as I was learning how the models work, so it's specifically optimized for the concepts that were hard for me when I started out. PyTorch and numpy.
- **Make it fast:** yoctogpt runs purely on cpu with extremely slow, unoptimized code. Once the GPT mental model is hammered out, my free [CUDA kernel programming course (12 hrs)](https://youtu.be/86FAWCzIe_4?si=QJQ4tA1jY9HtpCyA) (500K+ views) goes deep on GPU performance. Requires fluency in C and Python. Almost all of AI infra and performance focus in the world resides in CUDA.
