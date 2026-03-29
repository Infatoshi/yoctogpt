# yoctogpt
> Generatively Pre-train a Transformer in 100 lines of python

`v1.py` is [micrograd](https://github.com/karpathy/micrograd) + [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) (inspo from [nanogpt](https://github.com/karpathy/nanoGPT)). there are no dependencies. simply run `python3 v1.py` or `uv run v1.py` to see a GPT learn a tongue twister from zero knowledge about the world (brain starts as random numbers)

I admire karpathy's teaching approach where we introduces enough such that the curiousity of the student can fill in the rest. He makes it very fun to learn (I started with nanogpt :D). This is why I simplified this gpt-style pre-training educational example to exactly 100 lines. It requires experience with python, but if you put in the hours and study each line in here, you will have a solid mental model for how current LLMs and GPTs work.

In order to achieve this level of simplicity, I specifically use a simple character-level tokenizer, a single causal attention head rather than Multi-head Attention, a basic relu MLP (not MoE), a compact RMSprop like optimizer to help the nn learn fast without scarying you with complexity, val loss with text streaming to make it "feel" more intuitive, a simple version of position embeddings, and scalar level autograd to introduce backpropagation (how neural nets learn) in the cleanest way. You get some core lines of code to understand very deeply, and nothing else.

I do have a whiteboard explanation of backprop for the simplest neural net -- a multi-layer perceptron on my youtube channel [here](https://youtu.be/0dbihoMRuyg?si=CvXDG6BC8khGcxoT). I made this to help teach myself how backprop works. I knew I understood it fully when I could explain to an audience on a whiteboard.

If you want to build a more modern mental model on how these transformers work at a bigger scale (million to billions of params), I recommend taking a look at this course I built on LLMs from scratch (inspo by nanogpt). It assumes very basic knowledge. I actually built [this course](https://youtu.be/UU1WVnMk4E8?si=RTKgM3YJNJSwsHz3) as I was learning how the models work, so I can confidently say you'll be able to connect the dots as I specifically optimized this course for concepts that were hard for me when I started out. You'll use pytorch and numpy.

It runs purely on cpu with extremely slow and unoptimized code. If this is an area of interest and you have this GPT mental model well hammered out, I created a free course on [CUDA kernel programming (12 hrs)](https://youtu.be/86FAWCzIe_4?si=QJQ4tA1jY9HtpCyA) that has accumulated 500K+ views. This is a much deeper course which requires fluency in C and Python. Almost all of AI infra and performance focus in the world resides in CUDA.

Why did I make this you ask? Well... most GPT codebases are practical before they are obvious.

This one goes the other direction with just one file, standard library only, hardcoded dataset (yes, our goal is to overfit -- educational reasons), hardcoded hparams, no checkpoints, no batching, no optimizer complexity, no torch/tensorflow/numpy abstractions, and no comments.

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
