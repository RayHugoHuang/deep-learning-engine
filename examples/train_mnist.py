import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dle.nn as nn
from dle.tensor import Tensor, no_grad
from dle.optim import AdamW
from dle.quant import quantize_dynamic
from dle.utils import summary, train_epoch, validate_model, weights_init, save_model, load_model
from dataset import DataLoader, load_mnist

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)
        self.apply(weights_init("kaiming"))

    def forward(self, x):
        x = self.bn1(self.conv1(x)).relu()
        x = self.bn2(self.conv2(x)).relu()
        return self.fc(x.reshape(x.shape[0], -1))

def main():
    print("=== Vision Classification: MNIST CNN (60,000 samples) ===")
    (x_tr, y_tr), (x_te, y_te) = load_mnist()
    train_loader = DataLoader(x_tr[:10000], y_tr[:10000], batch_size=64)
    test_loader = DataLoader(x_te[:1000], y_te[:1000], batch_size=64, shuffle=False)

    model = CNN(in_channels=1, num_classes=10)
    summary(model, train_loader)

    optimizer = AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_epoch(model, train_loader, loss_fn, optimizer, epoch=0, epochs=1)
    validate_model(model, test_loader, mode="classification")

    # Delivery: Checkpoint persistence, quantization & sample prediction
    ckpt_path = "mnist_cnn.pth"
    save_model(model, ckpt_path)
    load_model(model, ckpt_path)
    if os.path.exists(ckpt_path): os.remove(ckpt_path)

    quantized = quantize_dynamic(model)
    with no_grad():
        preds = quantized(Tensor(x_te[:5])).data.argmax(axis=-1)
        print(f"Sample Predictions: {preds.tolist()} | Actual Digits: {y_te[:5].tolist()}\n")

if __name__ == "__main__":
    main()
