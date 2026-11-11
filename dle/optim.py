import sys
import numpy as np
from dle.tensor import Tensor

class Optimizer:
    def __init__(self, params):
        # Store parameter references
        self.params = list(params)

    def zero_grad(self):
        # Clear gradients across all tracked parameters
        for p in self.params:
            p.zero_grad()

    def step(self):
        # Must be overridden by subclasses (e.g. SGD, Adam)
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                g = p.grad + self.weight_decay * p.data
                self.m[i] = self.momentum * self.m[i] + g
                p.data -= self.lr * self.m[i]

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-5, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        step_size = self.lr / (1.0 - b1 ** self.t)
        inv_bc2 = 1.0 / (1.0 - b2 ** self.t)
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.m[i] = b1 * self.m[i] + (1.0 - b1) * p.grad
                self.v[i] = b2 * self.v[i] + (1.0 - b2) * (p.grad * p.grad)
                p.data -= step_size * self.m[i] / (np.sqrt(self.v[i] * inv_bc2) + self.eps)
                if self.weight_decay:
                    p.data -= self.lr * self.weight_decay * p.data

class AdamW(Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-5, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay=weight_decay)

class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()

class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Math: lr = base_lr * gamma^(last_epoch // step_size)
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=1e-5, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Math: lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * last_epoch / T_max))
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1.0 + np.cos(np.pi * self.last_epoch / self.T_max)
        )

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Math: lr = base_lr * gamma^last_epoch
        return self.base_lr * (self.gamma ** self.last_epoch)

lr_scheduler = sys.modules[__name__]

if __name__ == "__main__":
    # 1. Pipeline 1: Optimizer Convergence (SGD & Adam) & zero_grad Lifecycle
    for opt_cls, w_init, target, kwargs, steps in [
        (SGD, 5.0, 0.0, {"lr": 0.05, "momentum": 0.9}, 100),
        (Adam, 0.0, 3.0, {"lr": 0.1}, 150),
        (AdamW, 0.0, 3.0, {"lr": 0.1, "weight_decay": 0.01}, 150)
    ]:
        w = Tensor([w_init], requires_grad=True)
        opt = opt_cls([w], **kwargs)
        for _ in range(steps):
            opt.zero_grad()
            ((w - target) ** 2).sum().backward()
            opt.step()
        assert abs(w.data[0] - target) < 1e-1
        assert w.grad is not None
        opt.zero_grad()
        assert w.grad is None

    # 2. Pipeline 2: Learning Rate Schedulers Verification
    opt = SGD([Tensor(0.0)], lr=0.1)
    s_step = StepLR(opt, step_size=2, gamma=0.5)
    s_step.step(); s_step.step()
    assert np.isclose(opt.lr, 0.05)

    opt.lr = 0.1
    s_cos = CosineAnnealingLR(opt, T_max=4, eta_min=0.01)
    for _ in range(4): s_cos.step()
    assert np.isclose(opt.lr, 0.01)

    opt.lr = 0.1
    s_exp = ExponentialLR(opt, gamma=0.9)
    s_exp.step()
    assert np.isclose(opt.lr, 0.09)

    print("All optim.py optimizer & scheduler pipelines verified successfully!")
