import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dle.nn as nn
from dle.tensor import Tensor, no_grad
from dle.optim import AdamW, CosineAnnealingLR
from dle.utils import summary, train_epoch, validate_model, weights_init, save_model, load_model
from dataset import DataLoader, load_sine_wave

class SineLSTM(nn.Module):
    def __init__(self, in_features=1, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.apply(weights_init("xavier"))

    def forward(self, x):
        state = None
        for t in range(x.shape[1]):
            h, c = self.lstm(x[:, t], state)
            state = (h, c)
        return self.fc(h)

def main():
    print("=== Sequence Forecasting: Sine Wave LSTM (5,000 steps) ===")
    (x_tr, y_tr), (x_te, y_te) = load_sine_wave(num_samples=5000, seq_len=15)
    train_loader = DataLoader(x_tr, y_tr, batch_size=32)
    test_loader = DataLoader(x_te, y_te, batch_size=32, shuffle=False)

    model = SineLSTM(in_features=1, hidden_size=16)
    summary(model, train_loader)

    optimizer = AdamW(model.parameters(), lr=2e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        train_epoch(model, train_loader, loss_fn, optimizer, epoch=epoch, epochs=10, scheduler=scheduler)
    validate_model(model, test_loader, mode="regression")

    # Delivery: Checkpoint persistence & sample prediction
    ckpt_path = "sine_lstm.pth"
    save_model(model, ckpt_path)
    load_model(model, ckpt_path)
    if os.path.exists(ckpt_path): os.remove(ckpt_path)

    with no_grad():
        preds = model(Tensor(x_te[:3])).data.reshape(-1)
        print("Sample Forecasts (Pred vs Actual):")
        for p, a in zip(preds, y_te[:3].reshape(-1)):
            print(f"  Pred: {p:+.4f} | Actual: {a:+.4f}")
        print()

if __name__ == "__main__":
    main()
