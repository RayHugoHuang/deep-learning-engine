"""
Unified benchmark verification suite across all 4 deep learning domains:
1. Tabular Classification: Wine Quality MLP
2. Sequence Forecasting: Sine Wave LSTM
3. Vision Classification: MNIST CNN
4. Natural Language Processing: Tiny Shakespeare MiniGPT
"""
from examples.train_wine import main as run_wine
from examples.train_sine import main as run_sine
from examples.train_mnist import main as run_mnist
from examples.train_shakespeare import main as run_shakespeare

if __name__ == "__main__":
    run_wine()
    run_sine()
    run_mnist()
    run_shakespeare()
