import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dle.nn as nn
from dle.tensor import Tensor, no_grad
from dle.optim import AdamW, CosineAnnealingLR
from dle.quant import quantize_dynamic
from dle.utils import summary, train_epoch, validate_model, weights_init, save_model, load_model
from dataset import DataLoader, load_wine

class MLP(nn.Module):
    def __init__(self, in_features=11, hidden_features=64, num_classes=7):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes)
        self.apply(weights_init("kaiming"))

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())

def main():
    print("=== Tabular Classification: Wine Quality MLP (6,497 samples) ===")
    (x_tr, y_tr), (x_te, y_te) = load_wine()
    train_loader = DataLoader(x_tr, y_tr, batch_size=32)
    test_loader = DataLoader(x_te, y_te, batch_size=32, shuffle=False)

    model = MLP(in_features=11, hidden_features=64, num_classes=7)
    summary(model, train_loader)

    optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        train_epoch(model, train_loader, loss_fn, optimizer, epoch=epoch, epochs=10, scheduler=scheduler)
    validate_model(model, test_loader, mode="classification")

    # Delivery: Checkpoint persistence, quantization & sample prediction
    ckpt_path = "wine_mlp.pth"
    save_model(model, ckpt_path)
    load_model(model, ckpt_path)
    if os.path.exists(ckpt_path): os.remove(ckpt_path)

    quantized = quantize_dynamic(model)
    with no_grad():
        sample_preds = quantized(Tensor(x_te[:3])).data.argmax(axis=-1)
        print(f"Sample Predictions: {sample_preds.tolist()} | Targets: {y_te[:3].tolist()}\n")

if __name__ == "__main__":
    main()
