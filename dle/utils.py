import pickle, math
import numpy as np
import dle.nn as nn
from tqdm import tqdm
from dle.tensor import Tensor, no_grad

def save_model(model: nn.Module, savename: str):
    with open(savename if savename.endswith(".pth") else f"{savename}.pth", "wb") as f:
        pickle.dump({"model": model.state_dict()}, f)

def load_model(model: nn.Module, weight_name: str):
    with open(weight_name if weight_name.endswith(".pth") else f"{weight_name}.pth", "rb") as f:
        checkpoint = pickle.load(f)
    model.load_state_dict(checkpoint["model"], strict=False)

def clip_grad_norm_(parameters, max_norm, norm_type=2.0, eps=1e-6):
    # In-place gradient norm clipping across an iterable of Tensors
    params = [p for p in parameters if getattr(p, "grad", None) is not None]
    if not params:
        return 0.0
    if norm_type == float("inf"):
        total_norm = max(np.max(np.abs(p.grad)) for p in params)
    else:
        total_norm = np.power(sum(np.sum(np.abs(p.grad) ** norm_type) for p in params), 1.0 / norm_type)
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in params:
            p.grad *= clip_coef
    return float(total_norm)

def clip_grad_value_(parameters, clip_value):
    # In-place element-wise gradient value clipping across an iterable of Tensors
    for p in parameters:
        if getattr(p, "grad", None) is not None:
            p.grad = np.clip(p.grad, -clip_value, clip_value)

def count_parameters(model, trainable_only=True):
    # Count total number of scalar elements in a model or iterable of Tensors
    params = model.parameters() if hasattr(model, "parameters") else model
    return sum(p.data.size for p in params if not trainable_only or getattr(p, "requires_grad", True))

def accuracy(pred, target):
    # Compute accuracy from predictions (logits, probabilities, or labels) and target
    p = pred.data if hasattr(pred, "data") else np.asarray(pred)
    t = target.data if hasattr(target, "data") else np.asarray(target)
    if p.ndim > 1 and p.shape[-1] > 1:
        p = p.argmax(axis=-1)
    return float(np.mean(p == t))

def evaluate(pred, target):
    # Compute accuracy from predictions (Scalar or vector) and target (Scalar or vector)
    p = np.asarray(pred.data if hasattr(pred, "data") else pred).reshape(-1)
    t = np.asarray(target.data if hasattr(target, "data") else target).reshape(-1)
    r2_score = 1.0 - np.sum((t - p) ** 2) / (np.sum((t-t.mean()) ** 2) + 1e-8)
    mae = np.mean(np.abs(t - p))
    rmse = np.sqrt(np.mean((t - p) ** 2))
    mask = np.abs(t) > 1e-4
    mape = np.mean(np.abs((t[mask] - p[mask]) / (np.abs(t[mask]) + 1e-5))) * 100
    return {"r2": r2_score, "mae": mae, "rmse": rmse, "mape": mape}

def train_epoch(model, loader, loss_fn, opt, epoch=0, epochs=1, max_norm=5.0, scheduler=None):
    # Standard single epoch training runner with gradient clipping & scheduler support
    model.train()
    params = model.parameters()
    total_loss, count = 0.0, 0
    custom_format = "{desc}: {percentage:3.0f}% | {elapsed}<{remaining} | {rate_fmt} | {postfix}"
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", bar_format=custom_format)
    for x, y in pbar:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        if max_norm is not None:
            clip_grad_norm_(params, max_norm=max_norm)
        opt.step()
        total_loss += loss.item()
        count += 1
        pbar.set_postfix_str(f"Loss: {total_loss / count:.4f}")

    if scheduler is not None:
        scheduler.step()
    return total_loss / count if count > 0 else 0.0

@no_grad()
def validate_model(model, loader, mode="classification"):
    assert mode.lower() in ["classification", "regression"], "Invalid mode"
    model.eval()
    if mode.lower() == "classification":
        correct, total = 0.0, 0
        for x, y in loader:
            n = len(y.data)
            correct += accuracy(model(x), y) * n
            total += n
        print(f"Accuracy: {correct / total:.2%}\n")
    elif mode.lower() == "regression":
        preds, targets = zip(*[(model(x).data, y.data) for x, y in loader])
        m = evaluate(np.concatenate(preds), np.concatenate(targets))
        print(f"Regression: R2={m['r2']:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, MAPE={m['mape']:.2f}%\n")

def weights_init(init_type="gaussian"):
    # Factory returning module parameter initialization callback
    init_map = {
        "gaussian": lambda w: nn.init.normal_(w, 0.0, 0.02),
        "xavier": lambda w: nn.init.xavier_normal_(w, gain=math.sqrt(2)),
        "kaiming": lambda w: nn.init.kaiming_normal_(w, a=0),
        "default": lambda w: None,
    }
    assert init_type in init_map, f"Unsupported initialization: {init_type}"
    def init_fun(m):
        if m.__class__.__name__.startswith(("Conv", "Linear")) and hasattr(m, "weight"):
            init_map[init_type](m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun

def summary(model: nn.Module, input_size, batch_size=2, verbose=True):
    # Model architecture inspector and memory estimator similar to torchinfo
    x = input_size if isinstance(input_size, Tensor) else next(iter(input_size))[0] if hasattr(input_size, "batch_size") else Tensor(np.zeros((batch_size, *input_size)))
    records, orig_fwd, seen = [], {}, set()

    def _fmt(obj):
        if isinstance(obj, Tensor): return str(list(obj.shape))
        if isinstance(obj, (tuple, list)):
            s = [list(o.shape) for o in obj if isinstance(o, Tensor)]
            return str(s[0]) if len(s) == 1 else "(" + ", ".join(str(x) for x in s) + ")"
        return "--"

    # Hook all leaf modules (modules without child modules)
    def _hook(mod, name=""):
        children = [m for _, m in mod._members() if isinstance(m, nn.Module)]
        if not children:
            orig_fwd[mod] = mod.forward
            def hooked(*args, **kwargs):
                out = orig_fwd[mod](*args, **kwargs)
                if mod not in seen:
                    seen.add(mod)
                    inp = args[0] if len(args) > 0 else kwargs.get("x", None)
                    p = sum(param.data.size for param in mod.parameters(False))
                    elem = out.data.size if isinstance(out, Tensor) else sum(o.data.size for o in out if isinstance(o, Tensor))
                    records.append({
                        "desc": f"{mod.__class__.__name__}: {name or mod.__class__.__name__}",
                        "in_shape": _fmt(inp),
                        "out_shape": _fmt(out),
                        "params": p,
                        "elem": elem
                    })
                return out
            mod.forward = hooked
        for n, m in mod._members(name):
            if isinstance(m, nn.Module) and m is not mod:
                _hook(m, n)

    _hook(model)
    try:
        with no_grad():
            model(x)
    finally:
        for m, fn in orig_fwd.items():
            m.forward = fn

    p_tot = sum(p.data.size for p in model.parameters(False))
    p_train = sum(p.data.size for p in model.parameters(True))
    sz = [x.data.size * 4 / 1e6, sum(r["elem"] for r in records) * 8 / 1e6, p_tot * 4 / 1e6]

    sep = "=" * 85
    lines = [sep, f"{'Layer (type:name)':<28}{'Input Shape':<21}{'Output Shape':<21}{'Param #':<15}", sep]
    lines += [f"{r['desc']:<28}{r['in_shape']:<21}{r['out_shape']:<21}{r['params']:<15,}" for r in records]
    lines += [sep, f"Total params: {p_tot:,} | Trainable: {p_train:,} | Non-trainable: {p_tot - p_train:,}", "-" * 85]
    lines += [f"Memory (MB): Input={sz[0]:.2f}, Activation={sz[1]:.2f}, Params={sz[2]:.2f}, Total={sum(sz):.2f}", sep]

    summary_str = "\n".join(lines)
    if verbose:
        print(summary_str)
    return {"total_params": p_tot, "trainable_params": p_train, "records": records, "summary_str": summary_str}

if __name__ == "__main__":
    # Pipeline 1: Gradient Clipping Verification (Norm & Value)
    p1 = Tensor([3.0, 4.0], requires_grad=True) # Norm = 5.0
    p1.grad = p1.data.copy()
    norm = clip_grad_norm_([p1], max_norm=2.5)
    assert np.isclose(norm, 5.0)
    assert np.isclose(np.linalg.norm(p1.grad), 2.5)

    p2 = Tensor([-5.0, 10.0], requires_grad=True)
    p2.grad = p2.data.copy()
    clip_grad_value_([p2], clip_value=3.0)
    assert np.allclose(p2.grad, [-3.0, 3.0])

    # Pipeline 2: Parameter Counting & Metric Verification
    p_train = Tensor(np.zeros((3, 4)), requires_grad=True)
    p_frozen = Tensor(np.zeros((2, 2)), requires_grad=False)
    assert count_parameters([p_train, p_frozen], trainable_only=True) == 12
    assert count_parameters([p_train, p_frozen], trainable_only=False) == 16

    logits = Tensor([[2.0, 0.5], [0.1, 1.8], [3.0, 1.0]])
    targets = Tensor([0, 1, 0])
    assert np.isclose(accuracy(logits, targets), 1.0)
    assert np.isclose(accuracy(Tensor([0, 0, 0]), targets), 2.0 / 3.0)

    # Evaluate regression metrics
    res = evaluate(Tensor([1.0, 2.0, 3.0]), Tensor([1.0, 2.0, 3.0]))
    assert np.isclose(res["r2"], 1.0) and np.isclose(res["mae"], 0.0) and np.isclose(res["rmse"], 0.0)

    # Pipeline 3: train_epoch Runner with StepLR Scheduler
    from dle.nn import Linear, MSELoss
    from dle.optim import SGD, StepLR
    m_test = Linear(2, 1)
    opt_test = SGD(m_test.parameters(), lr=0.1)
    sched_test = StepLR(opt_test, step_size=1, gamma=0.5)
    dummy_loader = [(Tensor([[1.0, 2.0]]), Tensor([[3.0]]))]
    avg_loss = train_epoch(m_test, dummy_loader, MSELoss(), opt_test, epoch=0, epochs=1, max_norm=1.0, scheduler=sched_test)
    assert isinstance(avg_loss, float) and avg_loss >= 0.0
    assert np.isclose(opt_test.lr, 0.05)

    # Pipeline 4: Model Checkpointing (save_model & load_model)
    import os
    test_pth = "temp_test_model.pth"
    save_model(m_test, test_pth)
    m_reload = Linear(2, 1)
    load_model(m_reload, test_pth)
    for k, v in m_test.state_dict().items():
        assert np.allclose(v, m_reload.state_dict()[k])
    if os.path.exists(test_pth):
        os.remove(test_pth)

    # Pipeline 5: weights_init & validate_model Verification
    from dle.nn import Conv2d
    c_test = Conv2d(1, 2, 3)
    c_test.apply(weights_init("gaussian"))
    c_test.apply(weights_init("xavier"))
    c_test.apply(weights_init("kaiming"))
    assert np.all(c_test.bias.data == 0.0)
    validate_model(m_test, dummy_loader, mode="regression")

    # Pipeline 6: Model Summary Verification
    from dle.nn import Sequential, MaxPool2d, Flatten
    s_model = Sequential(
        Conv2d(1, 16, 3, padding=1),
        MaxPool2d(2, 2),
        Flatten(),
        Linear(16 * 14 * 14, 10)
    )
    s_info = summary(s_model, (1, 28, 28), batch_size=2, verbose=True)
    assert s_info["total_params"] == 16 * 1 * 3 * 3 + 16 + (16 * 14 * 14 * 10 + 10)
    assert s_info["trainable_params"] == s_info["total_params"]
    assert len(s_info["records"]) == 4

    print("All utils.py utility pipelines verified successfully!")
