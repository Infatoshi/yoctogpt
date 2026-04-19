# fmt: off
import math
import random

random.seed(67)
txt = "Peter Piper picked a peck of pickled peppers.\nA peck of pickled peppers Peter Piper picked.\nIf Peter Piper picked a peck of pickled peppers,\nWhere's the peck of pickled peppers Peter Piper picked?\n"
characters, vocab_size, context_length, embed_dim, num_layers, learning_rate = sorted(set(txt)), len(set(txt)), 8, 16, 1, 0.01
train_data, val_data = (ids := [characters.index(c) for c in txt])[: int(len(ids) * 0.9)], ids[int(len(ids) * 0.9) :]

class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data, self.grad, self.children, self.local_grads = data, 0.0, children, local_grads
    def __add__(self, b):
        b = b if isinstance(b, Value) else Value(b)
        return Value(self.data + b.data, (self, b), (1, 1))
    def __mul__(self, b):
        b = b if isinstance(b, Value) else Value(b)
        return Value(self.data * b.data, (self, b), (b.data, self.data))
    def __pow__(self, n):
        return Value(self.data**n, (self,), (n * self.data ** (n - 1),))
    def __sub__(self, b):
        return self + (b * -1)
    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))
    def exp(self):
        return Value((e := math.exp(self.data)), (self,), (e,))
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def backward(self):
        topo, seen = [], set()
        def walk(n):
            n not in seen and (seen.add(n), [walk(c) for c in n.children], topo.append(n))
        walk(self)
        self.grad = 1.0
        [setattr(c, "grad", c.grad + g * n.grad) for n in reversed(topo) for c, g in zip(n.children, n.local_grads)]

def make_matrix(rows, cols):
    return [[Value(random.gauss(0, 0.08)) for _ in range(cols)] for _ in range(rows)]

weights = {"token_embed": make_matrix(vocab_size, embed_dim), "position_embed": make_matrix(context_length, embed_dim), "output_proj": make_matrix(vocab_size, embed_dim)}
for i in range(num_layers):
    weights |= {f"layer{i}.{k}": make_matrix(embed_dim, embed_dim) for k in ["query", "key", "value", "attn_out"]}
    weights[f"layer{i}.ff_up"], weights[f"layer{i}.ff_down"] = make_matrix(embed_dim * 4, embed_dim), make_matrix(embed_dim, embed_dim * 4)
all_parameters = [p for w in weights.values() for row in w for p in row]

def linear(x, w):
    return [sum((wi * xi for wi, xi in zip(row, x)), Value(0)) for row in w]

def normalize(x):
    mean_squared = sum((xi * xi for xi in x), Value(0)) * (1 / len(x))
    return [xi * ((mean_squared + 1e-5) ** -0.5) for xi in x]

def softmax(x):
    ex = [(xi - max(v.data for v in x)).exp() for xi in x]
    return [xi * (total**-1) for xi in ex] if (total := sum(ex, Value(0))) else []

def gpt_forward(token_id, position, key_cache, value_cache):
    x = normalize([a + b for a, b in zip(weights["token_embed"][token_id], weights["position_embed"][position])])
    for i in range(num_layers):
        residual, x = x, normalize(x)
        q, k, v = [linear(x, weights[f"layer{i}.{c}"]) for c in ["query", "key", "value"]]
        key_cache[i], value_cache[i] = key_cache[i] + [k], value_cache[i] + [v]
        attention_scores = [sum((q[j] * key_cache[i][t][j] for j in range(embed_dim)), Value(0)) * (embed_dim**-0.5) for t in range(len(key_cache[i]))]
        attention_weights = softmax(attention_scores)
        attended = [sum((attention_weights[t] * value_cache[i][t][j] for t in range(len(value_cache[i]))), Value(0)) for j in range(embed_dim)]
        x = [a + b for a, b in zip(linear(attended, weights[f"layer{i}.attn_out"]), residual)]
        ff_hidden = [h.relu() for h in linear(normalize(x), weights[f"layer{i}.ff_up"])]
        x = [a + b for a, b in zip(linear(ff_hidden, weights[f"layer{i}.ff_down"]), x)]
    return linear(x, weights["output_proj"])

def compute_loss(token_sequence):
    n = min(context_length, len(token_sequence) - 1)
    key_cache, value_cache, total_loss = [[] for _ in range(num_layers)], [[] for _ in range(num_layers)], Value(0)
    for position in range(n):
        total_loss = total_loss + (softmax(gpt_forward(token_sequence[position], position, key_cache, value_cache))[token_sequence[position + 1]].log() * -1)
    return total_loss * (1 / n)

def generate_text(prompt="Peter Piper "):
    token_ids = [characters.index(c) for c in prompt]
    print(prompt, end="", flush=True)
    for _ in range(50):
        key_cache, value_cache = [[] for _ in range(num_layers)], [[] for _ in range(num_layers)]
        for position, token in enumerate(token_ids[-context_length:]):
            logits = gpt_forward(token, position, key_cache, value_cache)
        next_token = random.choices(range(vocab_size), weights=[p.data for p in softmax(logits)])[0]
        print(characters[(token_ids := token_ids + [next_token])[-1]], end="", flush=True)
    print()

for step in range(2001):
    if step % 100 == 0:
        print(f"\nstep {step}: val {compute_loss(val_data[:context_length + 1]).data:.4f}") or generate_text()
    start_idx = random.randint(0, len(train_data) - context_length - 1)
    current_loss = compute_loss(train_data[start_idx : start_idx + context_length + 1])
    current_loss.backward()
    for p in all_parameters:
        p.m = 0.99 * getattr(p, "m", 0.0) + 0.01 * (p.grad**2)
        p.data, p.grad = p.data - learning_rate * p.grad / (p.m**0.5 + 1e-8), 0.0
