import numpy as np
from contextlib import ContextDecorator

def _unbroadcast(grad, shape):
    # Align gradient shape with input shape by summing over broadcasted dimensions
    if grad.shape == shape:
        return grad
    ndiff = grad.ndim - len(shape)
    if ndiff > 0:
        grad = grad.sum(axis=tuple(range(ndiff)))
    keep_axes = tuple(i for i, (g_d, s_d) in enumerate(zip(grad.shape, shape)) if s_d == 1 and g_d != 1)
    return grad.sum(axis=keep_axes, keepdims=True) if keep_axes else grad

class no_grad(ContextDecorator):
    _enabled = True

    def __enter__(self):
        self.prev = no_grad._enabled
        no_grad._enabled = False
        return self

    def __exit__(self, *args):
        no_grad._enabled = self.prev

class Tensor:
    __hash__ = object.__hash__
    _amp_enabled = False
    _amp_dtype = np.float16

    def __init__(self, data, requires_grad=False):
        self.data = data if isinstance(data, np.ndarray) else np.asarray(data)  # Store tensor value
        self.requires_grad = bool(requires_grad and no_grad._enabled)           # Whether to compute gradient
        self.grad = None                                                        # Accumulated gradient
        self._prev_set = set()                                                  # Previous tensors in computation graph
        self._backward_fn = lambda: None                                        # Chain-rule backward function

    @property
    def _prev(self):
        return self._prev_set

    @_prev.setter
    def _prev(self, val):
        if no_grad._enabled and self.requires_grad:
            self._prev_set = set(val)
        else:
            self._prev_set = set()
            self.requires_grad = False

    @property
    def _backward(self):
        return self._backward_fn

    @_backward.setter
    def _backward(self, fn):
        if no_grad._enabled and self.requires_grad:
            self._backward_fn = fn
        else:
            self._backward_fn = lambda: None

    def _accumulate_grad(self, grad):
        # Unify gradient shape restoration and accumulation to eliminate redundancy
        g = _unbroadcast(grad, self.data.shape)
        self.grad = g if self.grad is None else self.grad + g

    def zero_grad(self):
        # Reset gradient to None for next optimization iteration
        self.grad = None

    def item(self):
        return self.data.item()

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self.requires_grad or other.requires_grad))
        out._prev = {self, other}

        def _backward():
            # Math: z = x + y => dL/dx = dL/dz, dL/dy = dL/dz
            if self.requires_grad:
                self._accumulate_grad(out.grad)
            if other.requires_grad:
                other._accumulate_grad(out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self.requires_grad or other.requires_grad))
        out._prev = {self, other}

        def _backward():
            # Math: z = x - y => dL/dx = dL/dz * (+1), dL/dy = dL/dz * (-1)
            if self.requires_grad:
                self._accumulate_grad(out.grad)
            if other.requires_grad:
                other._accumulate_grad(-out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self.requires_grad or other.requires_grad))
        out._prev = {self, other}

        def _backward():
            # Math: z = x * y => dL/dx = dL/dz * y, dL/dy = dL/dz * x
            if self.requires_grad:
                self._accumulate_grad(out.grad * other.data)
            if other.requires_grad:
                other._accumulate_grad(out.grad * self.data)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self.requires_grad or other.requires_grad))
        out._prev = {self, other}

        def _backward():
            # Math: z = x / y => dL/dx = dL/dz / y, dL/dy = -dL/dz * x / (y^2)
            if self.requires_grad:
                self._accumulate_grad(out.grad / other.data)
            if other.requires_grad:
                other._accumulate_grad(-out.grad * self.data / (other.data ** 2))

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return Tensor(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only supporting int/float powers"
        out = Tensor(self.data ** p, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = x^p => dy/dx = p * x^(p-1) => dL/dx = dL/dy * p * x^(p-1)
            if self.requires_grad:
                self._accumulate_grad(out.grad * p * (self.data ** (p - 1)))

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if Tensor._amp_enabled:
            dt = Tensor._amp_dtype
            out = Tensor((self.data.astype(dt) @ other.data.astype(dt)).astype(np.float32), (self.requires_grad or other.requires_grad))
        else:
            out = Tensor(self.data @ other.data, (self.requires_grad or other.requires_grad))
        out._prev = {self, other}

        def _backward():
            # Math: Z = X @ Y => dL/dX = dL/dZ @ Y.T, dL/dY = X.T @ dL/dZ
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.swapaxes(-1, -2))
            if other.requires_grad:
                if other.data.ndim == 2 and self.data.ndim > 2:
                    other._accumulate_grad(self.data.reshape(-1, self.data.shape[-1]).T @ out.grad.reshape(-1, out.grad.shape[-1]))
                else:
                    other._accumulate_grad(self.data.swapaxes(-1, -2) @ out.grad)

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: slice extraction => accumulate upstream gradient directly into indexed positions
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros(self.data.shape, dtype=out.grad.dtype)
                if isinstance(idx, (int, slice, type(Ellipsis))) or (
                    isinstance(idx, tuple) and all(isinstance(i, (int, slice, type(Ellipsis), type(None))) for i in idx)
                ):
                    self.grad[idx] += out.grad
                else:
                    np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = sum(x) => dy/dx_i = 1 => dL/dx_i = broadcast(dL/dy)
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    axes = [axis] if isinstance(axis, int) else axis
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)
                self._accumulate_grad(np.broadcast_to(grad, self.data.shape))

        out._backward = _backward
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = |x| => dy/dx = sign(x) => dL/dx = dL/dy * sign(x)
            if self.requires_grad:
                self._accumulate_grad(out.grad * np.sign(self.data))

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = mean(x) => dy/dx_i = 1/n => dL/dx_i = broadcast(dL/dy) / n
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    axes = [axis] if isinstance(axis, int) else axis
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)
                n = self.data.size if axis is None else int(np.prod([self.data.shape[a] for a in ([axis] if isinstance(axis, int) else axis)]))
                self._accumulate_grad(np.broadcast_to(grad, self.data.shape) / n)

        out._backward = _backward
        return out

    def square(self):
        out = Tensor(self.data * self.data, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = x^2 => dy/dx = 2x => dL/dx = dL/dy * 2x
            if self.requires_grad:
                self._accumulate_grad(out.grad * 2.0 * self.data)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = max(0, x) => dy/dx = 1 if x > 0 else 0 => dL/dx = dL/dy * (x > 0)
            if self.requires_grad:
                self._accumulate_grad(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = tanh(x) => dy/dx = 1 - y^2 => dL/dx = dL/dy * (1 - y^2)
            if self.requires_grad:
                self._accumulate_grad(out.grad * (1.0 - out.data * out.data))

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = sigmoid(x) => dy/dx = y * (1 - y) => dL/dx = dL/dy * y * (1 - y)
            if self.requires_grad:
                self._accumulate_grad(out.grad * out.data * (1.0 - out.data))

        out._backward = _backward
        return out

    def silu(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(self.data * s, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = x * s => dy/dx = s + x * s * (1 - s) => dL/dx = dL/dy * dy/dx
            if self.requires_grad:
                self._accumulate_grad(out.grad * (s + self.data * s * (1.0 - s)))

        out._backward = _backward
        return out

    swish = silu

    def gelu(self):
        x = self.data
        x2 = x * x
        t = np.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x2))
        h = 0.5 * (1.0 + t)
        out = Tensor(x * h, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: dy/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * du/dx
            if self.requires_grad:
                self._accumulate_grad(out.grad * (h + 0.3989422804014327 * x * (1.0 - t * t) * (1.0 + 0.134145 * x2)))

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data
        z_max = np.max(z, axis=axis, keepdims=True)
        exp_z = np.exp(z - z_max)
        s = exp_z / np.sum(exp_z, axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: dL/dx = s * (dL/ds - sum(dL/ds * s, axis, keepdims=True))
            if self.requires_grad:
                sum_grad_s = np.sum(out.grad * s, axis=axis, keepdims=True)
                self._accumulate_grad(s * (out.grad - sum_grad_s))

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = exp(x) => dy/dx = exp(x) = y => dL/dx = dL/dy * y
            if self.requires_grad:
                self._accumulate_grad(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = log(x) => dy/dx = 1/x => dL/dx = dL/dy / x
            if self.requires_grad:
                self._accumulate_grad(out.grad / self.data)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: Y = reshape(X) => dL/dX = reshape(dL/dY, X.shape)
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = axes[0]
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: Y = transpose(X, axes) => dL/dX = transpose(dL/dY, inv_axes)
            if self.requires_grad:
                if not axes:
                    self._accumulate_grad(out.grad.transpose())
                else:
                    inv_axes = np.argsort(axes)
                    self._accumulate_grad(out.grad.transpose(inv_axes))

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def squeeze(self, axis=None):
        # Math: Y = squeeze(X, axis) => dL/dX = reshape(dL/dY, X.shape)
        return self.reshape(np.squeeze(self.data, axis=axis).shape)

    def unsqueeze(self, axis):
        # Math: Y = unsqueeze(X, axis) => dL/dX = reshape(dL/dY, X.shape)
        return self.reshape(np.expand_dims(self.data, axis=axis).shape)

    def _reduce_extrema(self, fn, axis, keepdims):
        out_data = fn(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = max/min(x, axis) => dL/dx = dL/dy * (x == extremum) / ties_count
            if self.requires_grad:
                exp_d = np.expand_dims(out.data, axis=axis) if not keepdims and axis is not None else out.data
                exp_g = np.expand_dims(out.grad, axis=axis) if not keepdims and axis is not None else out.grad
                mask = (self.data == exp_d).astype(self.data.dtype)
                counts = np.sum(mask, axis=axis, keepdims=True)
                self._accumulate_grad(exp_g * (mask / counts))

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        return self._reduce_extrema(np.max, axis, keepdims)

    def min(self, axis=None, keepdims=False):
        return self._reduce_extrema(np.min, axis, keepdims)

    def clamp(self, min=None, max=None):
        out_data = np.clip(self.data, a_min=min, a_max=max)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            # Math: y = clamp(x, min, max) => dy/dx = 1 if min <= x <= max else 0 => dL/dx = dL/dy * mask
            if self.requires_grad:
                mask = np.ones_like(self.data, dtype=bool)
                if min is not None:
                    mask &= (self.data >= min)
                if max is not None:
                    mask &= (self.data <= max)
                self._accumulate_grad(out.grad * mask)

        out._backward = _backward
        return out

    clip = clamp

    __gt__ = lambda self, o: Tensor(self.data > (o.data if isinstance(o, Tensor) else o))
    __lt__ = lambda self, o: Tensor(self.data < (o.data if isinstance(o, Tensor) else o))
    __ge__ = lambda self, o: Tensor(self.data >= (o.data if isinstance(o, Tensor) else o))
    __le__ = lambda self, o: Tensor(self.data <= (o.data if isinstance(o, Tensor) else o))
    __eq__ = lambda self, o: Tensor(self.data == (o.data if isinstance(o, Tensor) else o))
    __ne__ = lambda self, o: Tensor(self.data != (o.data if isinstance(o, Tensor) else o))

    @staticmethod
    def _shape(shape):
        return shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape

    @staticmethod
    def zeros(*shape, requires_grad=False):
        return Tensor(np.zeros(Tensor._shape(shape)), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False):
        return Tensor(np.ones(Tensor._shape(shape)), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad=False):
        return Tensor(np.random.randn(*Tensor._shape(shape)), requires_grad=requires_grad)

    def backward(self, grad=None, retain_graph=False):
        if grad is None:
            grad = np.ones_like(self.data, dtype=float)
        self.grad = grad

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        for node in reversed(topo):
            node._backward()
            if not retain_graph:
                node._prev = set()
                node._backward = lambda: None

def cat(tensors, axis=0):
    # Concatenate list of Tensors along axis with reverse gradient slicing
    raw_data = [t.data if isinstance(t, Tensor) else np.asarray(t) for t in tensors]
    data = np.concatenate(raw_data, axis=axis)
    req_grad = any(isinstance(t, Tensor) and t.requires_grad for t in tensors)
    out = Tensor(data, requires_grad=req_grad)
    out._prev = set(t for t in tensors if isinstance(t, Tensor))

    def _backward():
        # Math: Y = cat([X_1, X_2, ...], axis) => dL/dX_i = slice(dL/dY, axis, start_i, end_i)
        idx = 0
        for t in tensors:
            size = t.data.shape[axis] if isinstance(t, Tensor) else np.asarray(t).shape[axis]
            slc = [slice(None)] * out.data.ndim
            slc[axis] = slice(idx, idx + size)
            if isinstance(t, Tensor) and t.requires_grad:
                t._accumulate_grad(out.grad[tuple(slc)])
            idx += size

    out._backward = _backward
    return out

concat = cat
Tensor.cat = staticmethod(cat)
Tensor.concat = staticmethod(cat)

if __name__ == "__main__":
    # 1. Pipeline 1: Composite DAG Forward, Autograd & Finite-Difference Grad Check
    def loss_graph(x1, x2, w):
        c = cat([x1, x2], axis=1)                          # (2, 4) - cat
        h1 = (c @ w).clamp(-2.0, 2.0)                      # (2, 3) - @, clamp
        h2 = h1.tanh() + h1.sigmoid() - h1.relu() + h1.gelu() - h1.silu() # activations (+ gelu, silu)
        h3 = (h2 * 2.0 / 3.0).abs() ** 2                   # (2, 3) - *, /, abs, pow
        h4 = h3.softmax(axis=-1).exp().log()               # (2, 3) - softmax, exp, log
        return (h4.max(axis=1).unsqueeze(1) + h4.min(axis=1, keepdims=True)).mean()

    np.random.seed(42)
    x1_val = np.random.randn(2, 2)
    x2_val = np.random.randn(2, 2)
    w_val = np.random.randn(4, 3)

    x1 = Tensor(x1_val.copy(), requires_grad=True)
    x2 = Tensor(x2_val.copy(), requires_grad=True)
    w = Tensor(w_val.copy(), requires_grad=True)

    loss = loss_graph(x1, x2, w)
    loss.backward()

    # Numerical gradient check across all inputs
    eps = 1e-5
    for t, val in [(x1, x1_val), (x2, x2_val), (w, w_val)]:
        num_grad = np.zeros_like(val)
        for idx in np.ndindex(val.shape):
            orig = val[idx]
            val[idx] = orig + eps
            lp = loss_graph(Tensor(x1_val if t is not x1 else val),
                            Tensor(x2_val if t is not x2 else val),
                            Tensor(w_val if t is not w else val)).data
            val[idx] = orig - eps
            lm = loss_graph(Tensor(x1_val if t is not x1 else val),
                            Tensor(x2_val if t is not x2 else val),
                            Tensor(w_val if t is not w else val)).data
            val[idx] = orig
            num_grad[idx] = (lp - lm) / (2 * eps)
        assert np.allclose(t.grad, num_grad, atol=1e-4, rtol=1e-3)
    print("Pipeline 1 (Composite DAG: Cat + Matmul + Activations + Reductions) verified!")

    # 2. Pipeline 2: Slicing, Transpose, Squeeze & Gradient Accumulation Lifecycle
    s = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    out2 = (s[:, :2].T.squeeze() + (-s[:, :2].T)).sum()
    out2.backward()
    assert np.allclose(s.grad, np.zeros_like(s.data))

    s.zero_grad()
    assert s.grad is None
    sl_part = s[:, 0:1]
    sl_part.backward()
    assert np.allclose(s.grad, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    print("Pipeline 2 (Tensor Transformations: Slicing + Transpose + Squeeze + Lifecycle) verified!")

    # 3. Pipeline 3: Comparisons, Factories & Scalar Properties
    t_comp = Tensor([[1.0, 5.0], [3.0, 2.0]])
    assert (t_comp > 2.0).data.tolist() == [[False, True], [True, False]]
    assert (t_comp <= 2.0).data.tolist() == [[True, False], [False, True]]
    assert (t_comp == 5.0).data.tolist() == [[False, True], [False, False]]
    assert (t_comp != 5.0).data.tolist() == [[True, False], [True, True]]
    assert Tensor.zeros(2, 3).shape == (2, 3) and Tensor.ones(2, 3).data.sum() == 6.0
    assert Tensor.randn(2, 3).shape == (2, 3) and Tensor(42.0).item() == 42.0

    # no_grad verification
    with no_grad():
        ng_out = s + 1.0
        assert ng_out.requires_grad is False
        assert len(ng_out._prev) == 0

    print("Pipeline 3 (Comparisons, Factories, no_grad & Scalar Properties) verified!")
