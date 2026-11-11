import numpy as np
from dle.tensor import Tensor
from dle.nn import Module, Linear, Sequential, ModuleList, ModuleDict

class QuantizedLinear(Module):
    def __init__(self, weight, scale_w, bias=None):
        super().__init__()
        self.weight, self.scale_w, self.bias = weight, float(scale_w), bias

    @classmethod
    def from_float(cls, mod: Linear):
        w = mod.weight.data
        scale_w = float(np.max(np.abs(w)) / 127.0) or 1.0
        w_int8 = np.clip(np.round(w / scale_w), -127, 127).astype(np.int8)
        bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
        return cls(w_int8, scale_w, bias)

    def forward(self, x):
        # Weight-only INT8 dequantization with native OpenBLAS Float32 GEMM acceleration
        arr = x.data if isinstance(x, Tensor) else np.asarray(x)
        out = arr @ (self.weight.astype(np.float32) * self.scale_w)
        if self.bias is not None:
            out += self.bias
        return Tensor(out)

def quantize_dynamic(model):
    if isinstance(model, Sequential):
        model.layers = [QuantizedLinear.from_float(c) if isinstance(c, Linear) else quantize_dynamic(c) for c in model.layers]
    elif isinstance(model, ModuleList):
        model.modules = [QuantizedLinear.from_float(c) if isinstance(c, Linear) else quantize_dynamic(c) for c in model.modules]
    elif isinstance(model, ModuleDict):
        model.modules = {k: QuantizedLinear.from_float(c) if isinstance(c, Linear) else quantize_dynamic(c) for k, c in model.modules.items()}
    elif isinstance(model, Module):
        for k, c in list(vars(model).items()):
            if isinstance(c, Linear): setattr(model, k, QuantizedLinear.from_float(c))
            elif isinstance(c, Module): quantize_dynamic(c)
    return model

if __name__ == "__main__":
    np.random.seed(42)
    # 1. Single Layer Quantization Verification
    lin = Linear(16, 8)
    q_lin = QuantizedLinear.from_float(lin)
    assert q_lin.weight.dtype == np.int8 and q_lin.weight.itemsize == 1
    x = Tensor(np.random.randn(4, 16))
    y_float, y_quant = lin(x).data, q_lin(x).data
    assert np.mean(np.abs(y_float - y_quant)) < 0.05

    # 2. End-to-End Model Dynamic Quantization Verification
    class CustomNet(Module):
        def __init__(self):
            super().__init__()
            self.seq = Sequential(Linear(16, 32))
            self.m_list = ModuleList([Linear(32, 16)])
            self.m_dict = ModuleDict({"head": Linear(16, 10)})
        def forward(self, x):
            return self.m_dict["head"](self.m_list[0](self.seq(x)))

    net = CustomNet()
    out_orig = net(x).data
    quantize_dynamic(net)
    assert isinstance(net.seq[0], QuantizedLinear) and isinstance(net.m_list[0], QuantizedLinear) and isinstance(net.m_dict["head"], QuantizedLinear)
    out_quant = net(x).data
    assert np.mean(np.abs(out_orig - out_quant)) < 0.05

    print("Quantization (QuantizedLinear + quantize_dynamic) verified successfully!")
