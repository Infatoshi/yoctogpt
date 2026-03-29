import math
import random

random.seed(67)
txt = "Peter Piper picked a peck of pickled peppers.\nA peck of pickled peppers Peter Piper picked.\nIf Peter Piper picked a peck of pickled peppers,\nWhere’s the peck of pickled peppers Peter Piper picked?\n"
chars, vocab, ctx, d, nl, lr = sorted(set(txt)), len(set(txt)), 8, 16, 1, 0.01
tr, va = (ids := [chars.index(c) for c in txt])[: int(len(ids) * 0.9)], ids[int(len(ids) * 0.9) :]

class V:
    def __init__(self, data, kids=(), grads=()):
        self.data = data
        self.grad = 0.0
        self.kids = kids
        self.grads = grads
    def __add__(self, b):
        b = b if isinstance(b, V) else V(b)
        return V(self.data + b.data, (self, b), (1, 1))
    def __mul__(self, b):
        b = b if isinstance(b, V) else V(b)
        return V(self.data * b.data, (self, b), (b.data, self.data))
    def __pow__(self, n):
        return V(self.data**n, (self,), (n * self.data ** (n - 1),))
    def __sub__(self, b):
        return self + (b * -1)
    def log(self):
        return V(math.log(self.data), (self,), (1 / self.data,))
    def exp(self):
        return V((e := math.exp(self.data)), (self,), (e,))
    def relu(self):
        return V(max(0, self.data), (self,), (float(self.data > 0),))
    def backward(self):
        topo, seen = [], set()
        def walk(n):
            n not in seen and (seen.add(n), [walk(k) for k in n.kids], topo.append(n))
        walk(self)
        self.grad = 1.0
        [setattr(k, "grad", k.grad + g * n.grad) for n in reversed(topo) for k, g in zip(n.kids, n.grads)]

def mat(r, c):
    return [[V(random.gauss(0, 0.08)) for _ in range(c)] for _ in range(r)]

sd = {"wte": mat(vocab, d), "wpe": mat(ctx, d), "lm": mat(vocab, d)}
for i in range(nl):
    sd |= {f"l{i}.{k}": mat(d, d) for k in "qkvo"}
    sd[f"l{i}.f1"], sd[f"l{i}.f2"] = mat(d * 4, d), mat(d, d * 4)
params = [p for w in sd.values() for row in w for p in row]

def lin(x, w):
    return [sum((wi * xi for wi, xi in zip(row, x)), V(0)) for row in w]

def norm(x):
    m = sum((xi * xi for xi in x), V(0)) * (1 / len(x))
    return [xi * ((m + 1e-5) ** -0.5) for xi in x]

def soft(xs):
    ex = [(x - max(v.data for v in xs)).exp() for x in xs]
    return [x * (tot**-1) for x in ex] if (tot := sum(ex, V(0))) else []

def gpt(tok, pos, ks, vs):
    x = norm([a + b for a, b in zip(sd["wte"][tok], sd["wpe"][pos])])
    for i in range(nl):
        r, x = x, norm(x)
        q, k, v = [lin(x, sd[f"l{i}.{c}"]) for c in "qkv"]
        ks[i], vs[i] = ks[i] + [k], vs[i] + [v]
        lgt = [sum((q[j] * ks[i][t][j] for j in range(d)), V(0)) * (d**-0.5) for t in range(len(ks[i]))]
        w = soft(lgt)
        att = [sum((w[t] * vs[i][t][j] for t in range(len(vs[i]))), V(0)) for j in range(d)]
        x = [a + b for a, b in zip(lin(att, sd[f"l{i}.o"]), r)]
        y = [xi.relu() for xi in lin(norm(x), sd[f"l{i}.f1"])]
        x = [a + b for a, b in zip(lin(y, sd[f"l{i}.f2"]), x)]
    return lin(x, sd["lm"])

def loss(xs):
    n = min(ctx, len(xs) - 1)
    ks, vs, z = [[] for _ in range(nl)], [[] for _ in range(nl)], V(0)
    for p in range(n):
        z = z + (soft(gpt(xs[p], p, ks, vs))[xs[p + 1]].log() * -1)
    return z * (1 / n)

def inference(prompt="Peter Piper "):
    toks = [chars.index(c) for c in prompt]
    print(prompt, end="", flush=True)
    for _ in range(50):
        ks, vs = [[] for _ in range(nl)], [[] for _ in range(nl)]
        for p, t in enumerate(toks[-ctx:]):
            lgt = gpt(t, p, ks, vs)
        tok = random.choices(range(vocab), weights=[p.data for p in soft(lgt)])[0]
        print(chars[(toks := toks + [tok])[-1]], end="", flush=True)
    print()

for s in range(501):
    if s % 100 == 0:
        v_l = loss(va[: ctx + 1]).data
        print(f"\nstep {s}: val {v_l:.4f}") or inference()
    idx = random.randint(0, len(tr) - ctx - 1)
    z = loss(tr[idx : idx + ctx + 1])
    z.backward()
    for p in params:
        p.m = 0.99 * getattr(p, "m", 0.0) + 0.01 * (p.grad**2)
        p.data, p.grad = p.data - lr * p.grad / (p.m**0.5 + 1e-8), 0.0
