import math
import numpy as np
from micrograd import Value


class Neuron:

    def __init__(self, nin, rng=None):

        self.nin = nin
        self.rng = rng if rng else np.random.default_rng(42)
        self.w =  [Value(self.rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(self.rng.uniform())

    def __call__(self, x):
        return (sum((i * j for i, j in zip(self.w, x)), self.b)).tanh()


class Layer:

    def __init__(self, nin, nout, rng=None):

        self.nin = nin
        self.nout = nout
        self.rng = rng if rng else np.random.default_rng(42)
        self.neurons = [Neuron(nin, self.rng) for _ in range(nout)]
        self.params = [w for n in self.neurons for w in n.w] + [n.b for n in self.neurons]

    def __call__(self, x):

        return [n(x) for n in self.neurons]


class MLP:

    def __init__(self, nin, nouts, lr=0.05):

        self.nin = nin
        self.nouts = nouts
        self.layers = [Layer(nin=nin, nout=nouts[0])] + [Layer(nouts[i], nouts[i+1]) for i in range(len(nouts) - 1)]
        self.params = [w for l in self.layers for w in l.params]
        self.lr = lr

    def __call__(self, x):

        stream = x
        for layer in self.layers:
            stream = layer(stream)

        return stream[0] if len(stream) < 2 else stream

    def __fit__(self, xs, ys):
        pass

    def step(self):
        for p in self.params:
            p.data = p.data - self.lr * p.grad


    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def mse_loss(self, ys_pred, ys):
        return sum(((y_pred - y)**2 for y_pred, y in zip(ys_pred, ys))) / len(ys)

    def fit(self, xs, ys, num_iters=50):

        for i in range(num_iters):
            ys_pred = [m(x) for x in xs]
            loss = self.mse_loss(ys_pred, ys)
            m.zero_grad()
            loss.backward()
            m.step()
            print("epoch:", i, "loss:", loss.data)



rng = np.random.default_rng(42)
nin = 3
ndata = 4


xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
print("y true", ys)
nouts = [4, 4, 1]
m = MLP(nin, nouts, lr=0.1)
ys_pred = []

m.fit(xs, ys)






