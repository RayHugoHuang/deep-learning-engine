import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from dle.tensor import Tensor

class Module:
    def __call__(self, *args, **kwargs):
        # Forward pass delegation
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # Must be overridden by subclasses
        raise NotImplementedError

    def _members(self, prefix=""):
        # Helper generator yielding (key, item) across direct attributes, lists, and dicts
        for name, val in self.__dict__.items():
            key = f"{prefix}{name}" if prefix == "" else f"{prefix}.{name}"
            if isinstance(val, (list, tuple)):
                for i, item in enumerate(val):
                    yield f"{key}.{i}", item
            elif isinstance(val, dict):
                for k, item in val.items():
                    yield f"{key}.{k}", item
            else:
                yield key, val

    def parameters(self, requires_grad_only=True):
        # Recursively collect Tensors from attributes, submodules, containers, and dicts
        params = []
        for _, item in self._members():
            if isinstance(item, Tensor) and (not requires_grad_only or item.requires_grad):
                params.append(item)
            elif isinstance(item, Module):
                params.extend(item.parameters(requires_grad_only))
        return params

    def zero_grad(self):
        # Clear gradients for all tracked parameters
        for p in self.parameters():
            p.zero_grad()

    def requires_grad_(self, requires_grad=True):
        # Set requires_grad flag across all parameters in self and submodules
        for p in self.parameters(requires_grad_only=False):
            p.requires_grad = requires_grad
        return self

    def freeze(self):
        # Freeze all parameters
        return self.requires_grad_(False)

    def unfreeze(self):
        # Unfreeze all parameters
        return self.requires_grad_(True)

    def train(self, mode=True):
        # Recursively set training mode for self, submodules, containers, and dicts
        self.training = mode
        for _, item in self._members():
            if isinstance(item, Module):
                item.train(mode)
        return self

    def eval(self):
        # Convenience method to switch model and submodules to evaluation mode
        return self.train(False)

    def apply(self, fn):
        # Recursively apply a function to all submodules and self
        for _, item in self._members():
            if isinstance(item, Module):
                item.apply(fn)
        fn(self)
        return self

    def state_dict(self, prefix=""):
        # Recursively collect parameter data dictionary for model checkpointing
        state = {}
        for key, item in self._members(prefix):
            if isinstance(item, Tensor):
                state[key] = item.data.copy()
            elif isinstance(item, Module):
                state.update(item.state_dict(prefix=key))
        return state

    def load_state_dict(self, state_dict, strict=True, prefix=""):
        # Recursively load parameter arrays from dictionary into model Tensors
        if prefix == "" and strict:
            model_keys = set(self.state_dict().keys())
            missing, unexpected = model_keys - set(state_dict.keys()), set(state_dict.keys()) - model_keys
            if missing or unexpected:
                raise KeyError(f"Error in load_state_dict: missing={missing}, unexpected={unexpected}")

        for key, item in self._members(prefix):
            if isinstance(item, Tensor):
                if key in state_dict:
                    val = np.asarray(state_dict[key])
                    if item.data.shape == val.shape:
                        item.data = val.copy()
                    elif strict:
                        raise ValueError(f"Shape mismatch for '{key}': expected {item.data.shape}, got {val.shape}")
                elif strict:
                    raise KeyError(f"Missing key in state_dict: '{key}'")
            elif isinstance(item, Module):
                item.load_state_dict(state_dict, strict=strict, prefix=key)
        return self

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers[0]) if len(layers) == 1 and isinstance(layers[0], (list, tuple)) else list(layers)

    def forward(self, x):
        # Forward pass cascading through sequential layers
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, idx):
        return Sequential(*self.layers[idx]) if isinstance(idx, slice) else self.layers[idx]

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def append(self, module):
        self.layers.append(module)
        return self

    def _members(self, prefix=""):
        for i, layer in enumerate(self.layers):
            yield f"{prefix}.{i}" if prefix else str(i), layer

class ModuleList(Module):
    def __init__(self, modules=None):
        self.modules = list(modules) if modules is not None else []

    def _members(self, prefix=""):
        for i, mod in enumerate(self.modules):
            yield f"{prefix}.{i}" if prefix else str(i), mod

    def __getitem__(self, idx):
        return ModuleList(self.modules[idx]) if isinstance(idx, slice) else self.modules[idx]

    def __setitem__(self, idx, module):
        self.modules[idx] = module

    def __delitem__(self, idx):
        del self.modules[idx]

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def append(self, module):
        self.modules.append(module)
        return self

    def extend(self, modules):
        self.modules.extend(modules)
        return self

    def insert(self, index, module):
        self.modules.insert(index, module)

    def pop(self, idx=-1):
        return self.modules.pop(idx)

    def clear(self):
        self.modules.clear()

class ModuleDict(Module):
    def __init__(self, modules=None):
        self.modules = dict(modules) if modules is not None else {}

    def _members(self, prefix=""):
        for k, mod in self.modules.items():
            yield f"{prefix}.{k}" if prefix else str(k), mod

    def __getitem__(self, key):
        return self.modules[key]

    def __setitem__(self, key, module):
        self.modules[key] = module

    def __delitem__(self, key):
        del self.modules[key]

    def __len__(self):
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def __contains__(self, key):
        return key in self.modules

    def keys(self):
        return self.modules.keys()

    def values(self):
        return self.modules.values()

    def items(self):
        return self.modules.items()

    def update(self, modules):
        self.modules.update(modules)
        return self

    def pop(self, key, *args):
        return self.modules.pop(key, *args)

    def clear(self):
        self.modules.clear()

class ReLU(Module): forward = lambda self, x: x.relu()
class Sigmoid(Module): forward = lambda self, x: x.sigmoid()
class Tanh(Module): forward = lambda self, x: x.tanh()
class GELU(Module): forward = lambda self, x: x.gelu()
class SiLU(Module): forward = lambda self, x: x.silu()
Swish = SiLU

class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return x.softmax(axis=self.axis)

class Flatten(Module):
    def forward(self, x):
        return x.reshape(x.data.shape[0], -1)

class LayerNorm(Module):
    def __init__(self, features, eps=1e-4):
        self.weight = Tensor(np.ones(features), requires_grad=True)
        self.bias = Tensor(np.zeros(features), requires_grad=True)
        self.features = features
        self.eps = eps

    def forward(self, x):
        # Shape: (..., features) [Channels-Last] -> (..., features)
        xc = x.data - x.data.mean(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt((xc * xc).mean(axis=-1, keepdims=True) + self.eps)
        normalized = xc * inv_std
        out = Tensor(normalized, requires_grad=x.requires_grad)
        out._prev = {x}

        def _backward():
            if x.requires_grad:
                # Math: dx = (inv_std / N) * (N * g - sum(g) - norm * sum(g * norm))
                N = x.data.shape[-1]
                g = out.grad
                x._accumulate_grad(inv_std / N * (N * g - g.sum(axis=-1, keepdims=True) - normalized * (g * normalized).sum(axis=-1, keepdims=True)))

        out._backward = _backward
        return out * self.weight + self.bias

class BatchNorm(Module):
    def __init__(self, features, eps=1e-4, momentum=0.1):
        self.training = True
        self.weight = Tensor(np.ones(features), requires_grad=True)
        self.bias = Tensor(np.zeros(features), requires_grad=True)
        self.features = features
        self.momentum = momentum
        self.running_mean = np.zeros(features)
        self.running_var = np.ones(features)
        self.eps = eps

    def forward(self, x):
        # Shape: (N, C, ...) [Channels-First] -> (N, C, ...)
        axes = (0,) + tuple(range(2, x.data.ndim))
        view_shape = (1, -1) + (1,) * (x.data.ndim - 2)
        if getattr(self, 'training', True):
            mean = x.data.mean(axis=axes, keepdims=True)
            xc = x.data - mean
            var = (xc * xc).mean(axis=axes, keepdims=True)
            self.running_mean = self.momentum * mean.ravel() + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var.ravel() + (1 - self.momentum) * self.running_var
            inv_std = 1.0 / np.sqrt(var + self.eps)
            normalized = xc * inv_std
        else:
            r_mean = self.running_mean.reshape(view_shape)
            r_var = self.running_var.reshape(view_shape)
            inv_std = 1.0 / np.sqrt(r_var + self.eps)
            normalized = (x.data - r_mean) * inv_std

        out = Tensor(normalized, requires_grad=x.requires_grad)
        out._prev = {x}

        def _backward():
            if x.requires_grad:
                # Math: dx = (inv_std / M) * (M * g - sum(g) - norm * sum(g * norm))
                if getattr(self, 'training', True):
                    M = x.data.size // x.data.shape[1]
                    g = out.grad
                    dx = inv_std / M * (M * g - g.sum(axis=axes, keepdims=True) - normalized * (g * normalized).sum(axis=axes, keepdims=True))
                else:
                    dx = out.grad * inv_std
                x._accumulate_grad(dx)

        out._backward = _backward
        w = self.weight.reshape(*view_shape)
        b = self.bias.reshape(*view_shape)
        return out * w + b

class Dropout(Module):
    def __init__(self, p=0.5, channel_wise=False):
        self.p, self.channel_wise = p, channel_wise
        self.training = True

    def forward(self, x):
        # Shape: (N, ...) [Element-wise or Channel-wise (N, C, ...)] -> (N, ...)
        if getattr(self, 'training', True):
            shape = (x.data.shape[0], x.data.shape[1]) + (1,) * (x.data.ndim - 2) if self.channel_wise else x.data.shape
            mask = (np.random.rand(*shape) >= self.p) / (1 - self.p)
            out = Tensor(x.data * mask, requires_grad=x.requires_grad)
            out._prev = {x}

            def _backward():
                if x.requires_grad:
                    x._accumulate_grad(out.grad * mask)

            out._backward = _backward
            return out

        return x  # In eval mode, return input unchanged

class Parameter(Tensor):
    def __init__(self, data=None, requires_grad=True):
        # Extract raw numpy array if input is already a Tensor
        d = data.data if isinstance(data, Tensor) else data
        super().__init__(d, requires_grad=requires_grad)

class Linear(Module):
    def __init__(self, input_features, output_features):
        self.weight = Parameter(np.random.randn(input_features, output_features) * 0.01)
        self.bias = Parameter(np.zeros(output_features))

    def forward(self, x):
        # Shape: (..., input_features) [Features-Last] -> (..., output_features)
        return x @ self.weight + self.bias

class RecurrentCell(Module):
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.w_ih = Parameter(np.random.randn(input_size, hidden_size) * 0.01)
        self.w_hh = Parameter(np.random.randn(hidden_size, hidden_size) * 0.01)
        self.bias = Parameter(np.zeros(hidden_size))

    def forward(self, x, h=None):
        # Shape: x=(N, input_size), h=(N, hidden_size) -> (N, hidden_size)
        if h is None:
            h = Tensor(np.zeros((x.data.shape[0], self.hidden_size)))
        return (x @ self.w_ih + h @ self.w_hh + self.bias).tanh()

class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.w_ih = Parameter(np.random.randn(input_size, 4 * hidden_size) * 0.01)
        self.w_hh = Parameter(np.random.randn(hidden_size, 4 * hidden_size) * 0.01)
        self.bias = Parameter(np.zeros(4 * hidden_size))

    @staticmethod
    def _activate_gates(gates, c, H):
        i = gates[:, 0 * H : 1 * H].sigmoid()
        f = gates[:, 1 * H : 2 * H].sigmoid()
        g = gates[:, 2 * H : 3 * H].tanh()
        o = gates[:, 3 * H : 4 * H].sigmoid()
        c_next = f * c + i * g
        return o * c_next.tanh(), c_next

    def forward(self, x, state=None):
        # Shape: x=(N, input_size), state=(h, c) with shape (N, hidden_size) -> (h, c)
        h, c = (Tensor(np.zeros((x.data.shape[0], self.hidden_size))), Tensor(np.zeros((x.data.shape[0], self.hidden_size)))) if state is None else state
        return self._activate_gates(x @ self.w_ih + h @ self.w_hh + self.bias, c, self.hidden_size)

class SlidingWindow(Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def _params(self, D):
        # Convert kernel_size, stride, padding to D-dimensional tuples
        to_tuple = lambda v: (v,) * D if isinstance(v, int) else tuple(v)
        return to_tuple(self.kernel_size), to_tuple(self.stride), to_tuple(self.padding)

    def unfold(self, x):
        # Sliding local patches unfolding across D spatial dimensions (im2col)
        data = x.data if isinstance(x, Tensor) else x
        D = data.ndim - 2
        K, S, P = self._params(D)
        pad_width = ((0, 0), (0, 0)) + tuple((p, p) for p in P)
        x_pad = np.pad(data, pad_width) if any(p > 0 for p in P) else data
        windowed = sliding_window_view(x_pad, window_shape=K, axis=tuple(range(2, 2 + D))) # im2col
        stride_slice = (slice(None), slice(None)) + tuple(slice(None, None, s) for s in S)
        return windowed[stride_slice]

    def fold(self, patches, target_shape):
        # Accumulate patches into spatial canvas and strip zero-padding (col2im)
        D = len(target_shape) - 2
        K, S, P = self._params(D)
        spatial = patches.shape[2 : 2 + D]
        canvas = np.zeros(target_shape[:2] + tuple(target_shape[2 + d] + 2 * P[d] for d in range(D)))
        for k_idx in np.ndindex(K):
            slices = (slice(None), slice(None)) + tuple(slice(k_idx[d], k_idx[d] + spatial[d] * S[d], S[d]) for d in range(D))
            canvas[slices] += patches[(slice(None), slice(None), Ellipsis) + k_idx]
        if any(p > 0 for p in P):
            unpad = (slice(None), slice(None)) + tuple(slice(P[d], -P[d] if P[d] > 0 else None) for d in range(D))
            return canvas[unpad]
        return canvas

    def forward(self, x):
        return self.unfold(x)

class ConvBase(SlidingWindow):
    def __init__(self, in_c, out_c, kernel_size, D, stride=1, padding=0, groups=1):
        super().__init__(kernel_size, stride, padding)
        assert in_c % groups == 0 and out_c % groups == 0, "Channels must be divisible by groups"
        self.groups = groups
        k = (kernel_size,) * D if isinstance(kernel_size, int) else tuple(kernel_size)
        self.weight = Parameter(np.random.randn(out_c, in_c // groups, *k) * 0.01)
        self.bias = Parameter(np.zeros(out_c))

    def forward(self, x):
        D = x.data.ndim - 2
        windows = self.unfold(x)
        G = self.groups
        Cin_g, Cout_g = windows.shape[1] // G, self.weight.data.shape[0] // G
        c_win, c_w = (1,) + tuple(range(2 + D, 2 + 2 * D)), tuple(range(1, 2 + D))

        out_temp = np.tensordot(windows, self.weight.data, axes=(c_win, c_w)) if G == 1 else np.concatenate([
            np.tensordot(windows[:, g * Cin_g : (g + 1) * Cin_g], self.weight.data[g * Cout_g : (g + 1) * Cout_g], axes=(c_win, c_w))
            for g in range(G)
        ], axis=-1)

        out = Tensor(out_temp.transpose((0, 1 + D) + tuple(range(1, 1 + D))) + self.bias.data.reshape((1, -1) + (1,) * D),
                     requires_grad=(x.requires_grad or self.weight.requires_grad or self.bias.requires_grad))
        out._prev = {x, self.weight, self.bias}

        def _backward():
            # Math: y_g = Conv(x_g, W_g) + b_g
            if self.bias.requires_grad:
                # dL/db = sum_{N, *spatial} (dL/dy)
                self.bias._accumulate_grad(out.grad.sum(axis=(0,) + tuple(range(2, 2 + D))))
            if self.weight.requires_grad:
                # dL/dW_g = tensordot(dL/dy_g, windows_g) over (N, *spatial) axes
                c_ax = (0,) + tuple(range(2, 2 + D))
                dw = np.tensordot(out.grad, windows, axes=(c_ax, c_ax)) if G == 1 else np.concatenate([
                    np.tensordot(out.grad[:, g * Cout_g : (g + 1) * Cout_g], windows[:, g * Cin_g : (g + 1) * Cin_g], axes=(c_ax, c_ax))
                    for g in range(G)
                ], axis=0)
                self.weight._accumulate_grad(dw)
            if x.requires_grad:
                # dL/dx = fold(concat_g(dL/dy_g @ W_g)) back to input spatial dimensions
                all_p = np.moveaxis(np.tensordot(out.grad, self.weight.data, axes=((1,), (0,))), 1 + D, 1) if G == 1 else np.concatenate([
                    np.moveaxis(np.tensordot(out.grad[:, g * Cout_g : (g + 1) * Cout_g], self.weight.data[g * Cout_g : (g + 1) * Cout_g], axes=((1,), (0,))), 1 + D, 1)
                    for g in range(G)
                ], axis=1)
                x._accumulate_grad(self.fold(all_p, x.data.shape))

        out._backward = _backward
        return out

class Conv1d(ConvBase):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super().__init__(in_c, out_c, kernel_size, 1, stride, padding, groups)

class Conv2d(ConvBase):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super().__init__(in_c, out_c, kernel_size, 2, stride, padding, groups)

class Conv3d(ConvBase):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super().__init__(in_c, out_c, kernel_size, 3, stride, padding, groups)

class ConvLSTM(Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, stride=1, padding=0):
        self.hidden_channels = hidden_channels
        self.conv_x = Conv2d(input_channels, 4 * hidden_channels, kernel_size, stride, padding)
        self.conv_h = Conv2d(hidden_channels, 4 * hidden_channels, kernel_size, stride, padding)

    def forward(self, x, state=None):
        # Shape: x=(N, C_in, H, W), state=(h, c) each (N, C_h, H, W) [Channels-First] -> (h, c)
        N, _, H, W = x.data.shape
        h, c = (Tensor(np.zeros((N, self.hidden_channels, H, W))), Tensor(np.zeros((N, self.hidden_channels, H, W)))) if state is None else state
        return LSTM._activate_gates(self.conv_x(x) + self.conv_h(h), c, self.hidden_channels)

class ConvTransposeBase(SlidingWindow):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, D=2):
        k = (kernel_size,) * D if isinstance(kernel_size, int) else kernel_size
        super().__init__(k, stride, padding)
        self.weight = Parameter(np.random.randn(in_channels, out_channels, *k) * 0.01)
        self.bias = Parameter(np.zeros(out_channels))

    def forward(self, x):
        D = x.data.ndim - 2
        K, S, P = self._params(D)
        N, _, *spatial_in = x.data.shape
        C_out = self.weight.data.shape[1]
        out_spatial = tuple((spatial_in[d] - 1) * S[d] + K[d] - 2 * P[d] for d in range(D))
        all_patches = np.moveaxis(np.tensordot(x.data, self.weight.data, axes=((1,), (0,))), 1 + D, 1)
        out_data = self.fold(all_patches, (N, C_out) + out_spatial) + self.bias.data.reshape((1, -1) + (1,) * D)

        out = Tensor(out_data, requires_grad=(x.requires_grad or self.weight.requires_grad or self.bias.requires_grad))
        out._prev = {x, self.weight, self.bias}

        def _backward():
            if self.bias.requires_grad:
                self.bias._accumulate_grad(out.grad.sum(axis=(0,) + tuple(range(2, 2 + D))))

            if self.weight.requires_grad or x.requires_grad:
                windows = self.unfold(out.grad)

            if self.weight.requires_grad:
                # Math: dW = sum(x * windows) contracted across batch and spatial positions
                contract_axes = (0,) + tuple(range(2, 2 + D))
                self.weight._accumulate_grad(np.tensordot(x.data, windows, axes=(contract_axes, contract_axes)))

            if x.requires_grad:
                # Math: dx = sum(windows * W) contracted over C_out and kernel spatial dimensions
                contract_win = (1,) + tuple(range(2 + D, 2 + 2 * D))
                contract_w = (1,) + tuple(range(2, 2 + D))
                dx_temp = np.tensordot(windows, self.weight.data, axes=(contract_win, contract_w))
                perm = (0, 1 + D) + tuple(range(1, 1 + D))
                x._accumulate_grad(dx_temp.transpose(perm))

        out._backward = _backward
        return out

class ConvTranspose1d(ConvTransposeBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, D=1)

class ConvTranspose2d(ConvTransposeBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, D=2)

class ConvTranspose3d(ConvTransposeBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, D=3)

class PoolBase(SlidingWindow):
    def __init__(self, kernel_size=2, stride=None, padding=0, mode="max", D=2):
        k = (kernel_size,) * D if isinstance(kernel_size, int) else kernel_size
        stride = k if stride is None else (stride,) * D if isinstance(stride, int) else stride
        super().__init__(k, stride, padding)
        self.mode = mode

    def forward(self, x):
        D = x.data.ndim - 2
        N, C = x.data.shape[:2]
        windows = self.unfold(x)
        out_spatial = tuple(windows.shape[2:2 + D])

        # Flatten kernel spatial dimensions to -1 so argmax can operate on a single axis
        flat_windows = windows.reshape(N, C, *out_spatial, -1)
        if self.mode == "max":
            out_data = flat_windows.max(axis=-1)
            argmax = flat_windows.argmax(axis=-1)
        else:
            out_data = flat_windows.mean(axis=-1)

        out = Tensor(out_data, requires_grad=x.requires_grad)
        out._prev = {x}

        def _backward():
            if x.requires_grad:
                K, *_ = self._params(D)
                k_size = int(np.prod(K))
                if self.mode == "max":
                    dpatch = np.zeros((N, C, *out_spatial, k_size))
                    np.put_along_axis(dpatch, argmax[..., None], out.grad[..., None], axis=-1)
                    dpatch = dpatch.reshape((N, C, *out_spatial) + K)
                else:
                    dpatch = np.broadcast_to((out.grad / k_size)[..., None], (N, C, *out_spatial, k_size)).reshape((N, C, *out_spatial) + K)
                x._accumulate_grad(self.fold(dpatch, x.data.shape))

        out._backward = _backward
        return out

class MaxPool1d(PoolBase):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__(kernel_size, stride, padding, mode="max", D=1)

class MaxPool2d(PoolBase):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__(kernel_size, stride, padding, mode="max", D=2)

class MaxPool3d(PoolBase):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__(kernel_size, stride, padding, mode="max", D=3)

class AveragePool1d(PoolBase):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__(kernel_size, stride, padding, mode="mean", D=1)

class AveragePool2d(PoolBase):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__(kernel_size, stride, padding, mode="mean", D=2)

class AveragePool3d(PoolBase):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__(kernel_size, stride, padding, mode="mean", D=3)

class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pos = np.arange(0, max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
        self.pe = pe[None, :, :]

    def forward(self, x):
        return x + Tensor(self.pe[:, :x.data.shape[1], :])

class RotaryEmbedding(Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        self.cos = np.cos(emb)[None, None, :, :]
        self.sin = np.sin(emb)[None, None, :, :]

    def forward(self, q, k=None):
        L, d = q.data.shape[2], self.dim // 2
        c, s = self.cos[:, :, :L, :], self.sin[:, :, :L, :]
        def _rot(x):
            out = Tensor(x.data * c + np.concatenate([-x.data[..., d:], x.data[..., :d]], axis=-1) * s, requires_grad=x.requires_grad)
            out._prev = {x}
            def _backward():
                if x.requires_grad:
                    g = out.grad
                    x._accumulate_grad(g * c + np.concatenate([g[..., d:], -g[..., :d]], axis=-1) * s)
            out._backward = _backward
            return out
        return (_rot(q), _rot(k)) if k is not None else _rot(q)

_MASK_CACHE = {}
def causal_mask(seq_len):
    # Cached upper triangular mask with -1e9 above diagonal for causal self-attention
    if seq_len not in _MASK_CACHE:
        _MASK_CACHE[seq_len] = Tensor(np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)[None, None, :, :])
    return _MASK_CACHE[seq_len]

class MultiHeadAttention(Module):
    def __init__(self, input_features, output_features, num_heads):
        assert input_features % num_heads == 0, "Input features must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = input_features // num_heads
        self.q = Linear(input_features, input_features)
        self.k = Linear(input_features, input_features)
        self.v = Linear(input_features, output_features)

    def forward(self, x, mask=None, rotary_emb=None):
        # Shape: (N, L, input_features) -> (N, L, output_features)
        N, L, _ = x.data.shape
        q = self.q(x).reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(N, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        if rotary_emb is not None:
            q, k = rotary_emb(q, k)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.head_dim ** -0.5)
        if mask is not None:
            scores = scores + mask
        out = (scores.softmax(axis=-1) @ v).transpose(0, 2, 1, 3).reshape(N, L, -1)
        return out

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim) * 0.01)

    def forward(self, x):
        idx = x.data if isinstance(x, Tensor) else np.asarray(x)
        out = Tensor(self.weight.data[idx], requires_grad=self.weight.requires_grad)
        out._prev = {self.weight}

        def _backward():
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = np.zeros_like(self.weight.data)
                np.add.at(self.weight.grad, idx, out.grad)

        out._backward = _backward
        return out
        
class _Loss(Module):
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def _reduce(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class MAELoss(_Loss):
    def forward(self, pred, target):
        return self._reduce((pred - target).abs())

class MSELoss(_Loss):
    def forward(self, pred, target):
        return self._reduce((pred - target).square())

class CrossEntropyLoss(_Loss):
    def forward(self, pred, target):
        z = pred.data
        z_max = np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z - z_max)
        sum_exp = np.sum(exp_z, axis=-1, keepdims=True)
        softmax = exp_z / sum_exp
        target_data = target.data if isinstance(target, Tensor) else np.asarray(target)
        if target_data.shape != z.shape:
            one_hot = np.zeros_like(z)
            np.put_along_axis(one_hot, np.expand_dims(target_data, -1), 1.0, axis=-1)
        else:
            one_hot = target_data

        log_softmax = (z - z_max) - np.log(sum_exp)
        loss_val = -np.sum(one_hot * log_softmax, axis=-1)
        out = Tensor(loss_val, requires_grad=pred.requires_grad)
        out._prev = {pred}

        def _backward():
            if pred.requires_grad:
                dloss = out.grad[..., None] if out.grad.ndim < softmax.ndim else out.grad
                pred._accumulate_grad(dloss * (softmax - one_hot))

        out._backward = _backward
        return self._reduce(out)

class Init:
    @staticmethod
    def zeros_(tensor):
        tensor.data.fill(0.0)
        return tensor

    @staticmethod
    def ones_(tensor):
        tensor.data.fill(1.0)
        return tensor
    
    @staticmethod
    def constant_(tensor, val):
        tensor.data.fill(val)
        return tensor

    @staticmethod
    def normal_(tensor, mean=0.0, std=1.0):
        tensor.data[...] = np.random.normal(mean, std, size=tensor.data.shape)
        return tensor

    @staticmethod
    def uniform_(tensor, a=0.0, b=1.0):
        tensor.data[...] = np.random.uniform(a, b, size=tensor.data.shape)
        return tensor

    @staticmethod
    def _calculate_fan_in_and_fan_out(tensor):
        shape = tensor.data.shape
        assert len(shape) >= 2, "Fan in and fan out requires at least 2 dimensions"
        if len(shape) == 2:  # Linear: (in_features, out_features)
            return shape[0], shape[1]
        # Conv: (out_channels, in_channels, *kernel_size)
        k = int(np.prod(shape[2:]))
        return shape[1] * k, shape[0] * k

    @staticmethod
    def xavier_uniform_(tensor, gain=1.0):
        # Glorot uniform: U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
        fan_in, fan_out = Init._calculate_fan_in_and_fan_out(tensor)
        a = gain * np.sqrt(6.0 / (fan_in + fan_out))
        return Init.uniform_(tensor, -a, a)

    @staticmethod
    def xavier_normal_(tensor, gain=1.0):
        # Glorot normal: N(0, std^2) where std = gain * sqrt(2 / (fan_in + fan_out))
        fan_in, fan_out = Init._calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return Init.normal_(tensor, 0.0, std)

    @staticmethod
    def kaiming_uniform_(tensor, a=0.0):
        # He uniform: U(-bound, bound) where bound = sqrt(6 / ((1 + a^2) * fan_in))
        fan_in, _ = Init._calculate_fan_in_and_fan_out(tensor)
        bound = np.sqrt(6.0 / ((1.0 + a ** 2) * fan_in))
        return Init.uniform_(tensor, -bound, bound)

    @staticmethod
    def kaiming_normal_(tensor, a=0.0):
        # He normal: N(0, std^2) where std = sqrt(2 / ((1 + a^2) * fan_in))
        fan_in, _ = Init._calculate_fan_in_and_fan_out(tensor)
        std = np.sqrt(2.0 / ((1.0 + a ** 2) * fan_in))
        return Init.normal_(tensor, 0.0, std)

init = Init

if __name__ == "__main__":
    np.random.seed(42)
    def grad_check(model, x, eps=1e-5):
        # Unified numerical gradient check for inputs and all learnable parameters
        model.zero_grad()
        if x.requires_grad: x.zero_grad()
        model(x).sum().backward()
        targets = ([x] if x.requires_grad else []) + list(model.parameters())
        ana_grads = [t.grad.copy() for t in targets]

        for t, ana_grad in zip(targets, ana_grads):
            num_grad = np.zeros_like(t.data)
            for idx in np.ndindex(t.data.shape):
                orig = t.data[idx]
                t.data[idx] = orig + eps; lp = model(x).data.sum()
                t.data[idx] = orig - eps; lm = model(x).data.sum()
                t.data[idx] = orig
                num_grad[idx] = (lp - lm) / (2 * eps)
            assert np.allclose(ana_grad, num_grad, atol=1e-4, rtol=1e-3)
        return True

    # Pipeline 1: 2D Multi-Task Vision (Sequential Backbone + Dropout + ModuleDict Heads)
    class VisionMultiTaskNet(Module):
        def __init__(self):
            self.backbone = Sequential(
                Conv2d(1, 2, kernel_size=3, padding=1),
                Conv2d(2, 2, kernel_size=3, padding=1, groups=2),
                BatchNorm(2),
                ReLU(),
                MaxPool2d(2, 2),
                Dropout(0.2),
                Flatten()
            )
            self.heads = ModuleDict({"clf": Linear(2 * 2 * 2, 2)})
            self.heads.update({"aux": Linear(2 * 2 * 2, 2)})

        def forward(self, x):
            feat = self.backbone(x)
            return self.heads["clf"](feat) + self.heads["aux"](feat)

    m1 = VisionMultiTaskNet()
    m1.apply(lambda m: (init.xavier_uniform_(m.weight), init.zeros_(m.bias)) if isinstance(m, Linear)
             else (init.kaiming_normal_(m.weight), init.constant_(m.bias, 0.5)) if isinstance(m, Conv2d) else None)
    assert np.all(m1.heads["clf"].bias.data == 0.0) and np.all(m1.backbone[0].bias.data == 0.5)
    t2d = Tensor(np.empty((4, 4)))
    init.xavier_normal_(t2d); init.kaiming_uniform_(t2d); init.ones_(t2d)
    assert np.all(t2d.data == 1.0)

    m1.eval()  # Disables Dropout for deterministic numerical grad_check
    x1 = Tensor(np.random.randn(2, 1, 4, 4), requires_grad=True)
    assert grad_check(m1, x1)
    assert len(m1.parameters()) == 10
    m1.train(True)
    assert m1.backbone[2].training is True and m1.heads["clf"].training is True
    sd1 = m1.state_dict()
    m1_loaded = VisionMultiTaskNet()
    m1_loaded.load_state_dict(sd1)
    for k, v in m1_loaded.state_dict().items():
        assert np.allclose(sd1[k], v)
    partial_sd = {k: v for k, v in sd1.items() if "clf" in k}
    m1_partial = VisionMultiTaskNet()
    m1_partial.load_state_dict(partial_sd, strict=False)
    for k, v in partial_sd.items():
        assert np.allclose(m1_partial.state_dict()[k], v)
    print("Pipeline 1 (2D Vision: Sequential + Depthwise Conv2d + Dropout + ModuleDict) verified!")

    # Pipeline 2: 3D Spatiotemporal (Conv3d + 5D BatchNorm + MaxPool3d + ConvLSTM + Grouped Conv1d)
    class VideoNet3D(Module):
        def __init__(self):
            self.conv3d = Conv3d(1, 2, kernel_size=2, padding=1)
            self.bn5d = BatchNorm(2)  # Tests multi-dimensional 5D BatchNorm
            self.pool3d = MaxPool3d(2, 2)
            self.clstm = ConvLSTM(2, 2, kernel_size=3, padding=1)
            self.conv1d = Conv1d(2, 2, kernel_size=2, groups=2)

        def forward(self, x5d):
            feat = self.pool3d(self.bn5d(self.conv3d(x5d)))
            h, _ = self.clstm(feat[:, :, 0])
            return self.conv1d(h.reshape(h.data.shape[0], 2, -1))

    m2 = VideoNet3D()
    x2 = Tensor(np.random.randn(1, 1, 2, 4, 4), requires_grad=True)
    assert grad_check(m2, x2)
    print("Pipeline 2 (3D Spatiotemporal: Conv3d + 5D BatchNorm + ConvLSTM) verified!")

    # Pipeline 3: Sequence & Modern Transformer (Embedding + PositionalEncoding + RoPE + Causal Attention + GELU + LSTM)
    class ModernSeqTransformer(Module):
        def __init__(self):
            self.emb = Embedding(10, 4)
            self.pe = PositionalEncoding(4)
            self.rope = RotaryEmbedding(2)
            self.attn = MultiHeadAttention(4, 4, 2)
            self.mlp = Sequential(Linear(4, 8), GELU(), Linear(8, 4))
            self.ln = LayerNorm(4)
            self.lstm = LSTM(4, 2)

        def forward(self, x):
            tok = self.pe(self.emb(x))
            mask = causal_mask(tok.data.shape[1])
            attn = self.attn(tok, mask=mask, rotary_emb=self.rope)
            out = self.ln(tok + self.mlp(attn))
            h_lstm, c_lstm = self.lstm(out[:, 0])
            return h_lstm + c_lstm

    m3 = ModernSeqTransformer()
    x3 = Tensor(np.array([[1, 2, 3, 4]]))
    assert grad_check(m3, x3)
    print("Pipeline 3 (Sequence: Embedding + RoPE + Causal Attention + GELU + LSTM) verified!")

    # Pipeline 4: Generative Reconstruction (Autoencoder: Conv2d + AveragePool2d + ConvTranspose2d + Sigmoid)
    class Autoencoder(Module):
        def __init__(self):
            self.encoder = Sequential(Conv2d(1, 2, kernel_size=2), AveragePool2d(2))
            self.decoder = Sequential(ConvTranspose2d(2, 1, kernel_size=2), Sigmoid())

        def forward(self, x):
            return self.decoder(self.encoder(x))

    m4 = Autoencoder()
    x4 = Tensor(np.random.randn(2, 1, 4, 4), requires_grad=True)
    assert grad_check(m4, x4)
    print("Pipeline 4 (Generative: ConvTranspose2d + AveragePool2d) verified!")

    # Parameterized Loss Functions Verification
    class LossWrapper(Module):
        def __init__(self, loss_fn, target):
            self.loss_fn, self.target = loss_fn, target
        def forward(self, pred):
            return self.loss_fn(pred, self.target)

    t3 = Tensor(np.random.randn(2, 3))
    t1 = Tensor(np.array([1, 0]))
    for fn, tgt in [(MSELoss(), t3), (MAELoss(), t3), (CrossEntropyLoss(), t1), (CrossEntropyLoss(reduction="sum"), t1)]:
        assert grad_check(LossWrapper(fn, tgt), Tensor(np.random.randn(2, 3), requires_grad=True))
    print("Loss functions (MSELoss, MAELoss, CrossEntropyLoss) verified with full grad_check!")
