import numpy as np
from contextlib import ContextDecorator
from dle.tensor import Tensor

class Autocast(ContextDecorator):
    def __init__(self, enabled=True, dtype=np.float16):
        self.enabled, self.target_dtype = enabled, dtype

    def __enter__(self):
        self.prev, self.prev_dtype = Tensor._amp_enabled, Tensor._amp_dtype
        Tensor._amp_enabled, Tensor._amp_dtype = self.enabled, self.target_dtype
        return self

    def __exit__(self, *args):
        Tensor._amp_enabled, Tensor._amp_dtype = self.prev, self.prev_dtype

autocast = Autocast

class GradScaler:
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self._scale, self.growth_factor = float(init_scale), growth_factor
        self.backoff_factor, self.growth_interval = backoff_factor, growth_interval
        self._growth_tracker, self._found_inf = 0, False

    def scale(self, loss):
        return loss * self._scale

    def step(self, optimizer):
        self._found_inf = False
        for p in optimizer.params:
            if p.grad is not None:
                p.grad /= self._scale
                if not np.isfinite(p.grad).all():
                    self._found_inf = True
                    return
        optimizer.step()

    def update(self):
        if self._found_inf:
            self._scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self._scale *= self.growth_factor
                self._growth_tracker = 0

if __name__ == "__main__":
    from dle.nn import Linear
    from dle.optim import SGD

    # 1. Test Autocast context and matmul execution
    assert not Tensor._amp_enabled
    with autocast(dtype=np.float16):
        assert Tensor._amp_enabled
        assert Tensor._amp_dtype == np.float16
        a = Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
        b = Tensor(np.array([[3.0], [4.0]], dtype=np.float32))
        c = a @ b
        assert c.data.dtype == np.float32
        assert c.data[0, 0] == 11.0
    assert not Tensor._amp_enabled

    # 2. Test GradScaler scaling & unscaling
    scaler = GradScaler(init_scale=1024.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2)
    m = Linear(2, 1)
    opt = SGD(m.parameters(), lr=0.1)

    # Normal step
    x = Tensor(np.random.randn(2, 2))
    opt.zero_grad()
    with autocast():
        loss = (m(x) ** 2).sum()
    scaler.scale(loss).backward()
    assert m.weight.grad is not None
    scaler.step(opt)
    scaler.update()
    assert scaler._scale == 1024.0

    # Trigger second clean step to test growth
    opt.zero_grad()
    with autocast():
        loss = (m(x) ** 2).sum()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    assert scaler._scale == 2048.0

    # Trigger overflow to test backoff & step skip
    opt.zero_grad()
    with autocast():
        loss = (m(x) ** 2).sum()
    scaler.scale(loss).backward()
    m.weight.grad[0, 0] = np.inf
    w_before = m.weight.data.copy()
    scaler.step(opt)
    assert np.all(m.weight.data == w_before)
    scaler.update()
    assert scaler._scale == 1024.0

    print("AMP (autocast + GradScaler) verified successfully!")
