import numpy as np
import math
import torch as T

class Value:

    def __init__(self, data, op='', ch = (), label=''):

        self.data = data
        self.op = op
        self.ch = ch
        self.label = label
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):

        return f"Value(data={self.data:.4f})"

    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, op='+', ch=(self, other))

        def _backward():
            self.grad +=  out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, op='*', ch=(self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):

        return self + -1 * other

    def __rsub__(self, other):

        return self + -1 * other

    def __pow__(self, other):

        assert isinstance(other, (float, int))
        out = Value(self.data ** other, op='pow', ch=(self,))
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad 

        out._backward = _backward
        return out

    def exp(self):

        out = Value(math.exp(self.data), op='exp', ch=(self,))
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):

        return (self.exp() - (-1 * self).exp())/(self.exp() + (-1 * self).exp())


        

    def __truediv__(self, other):

        return self * other**-1

    def __rtruediv__(self, other):
    
        return self * other**-1

    def backward(self):

        # get topo order 
        topo = []
        visited = set()
        def _get_topo_order(node):

            for c in node.ch:
                if c in visited:
                    continue
                _get_topo_order(c)

            topo.append(node)
            visited.add(node)

        _get_topo_order(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()
            


if __name__ == '__main__':

    a = Value(-1.0)
    b = Value(3.0)
    c = 8 * a + b
    # d = a ** 4 + 10
    # o = (c+d).tanh()
    o = c.tanh()
    # grad a = 8, grad b = 1
    # print(o.ch)

    o.grad = 1
    o.backward()
    print("grad a:", a.grad, "grad b", b.grad)

    a, b = T.tensor([-1.0], requires_grad=True, dtype=T.float64),  T.tensor([3.0], requires_grad=True, dtype=T.float64)
    c = 8 * a + b
    # d = a ** 4 + 10
    # o = T.tanh(c + d)
    o = T.tanh(c)

    o.backward()
    print("grad a:", a.grad.item(), "grad b", b.grad.item())