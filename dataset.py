import os
import urllib.request
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from dle.tensor import Tensor

class Dataset:
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, idx):
        raise NotImplementedError

def default_collate(batch):
    # Recursively collate list of sample tuples/tensors into batched Tensors
    if isinstance(batch[0], (tuple, list)):
        return tuple(default_collate(samples) for samples in zip(*batch))
    return Tensor(np.stack([b.data if isinstance(b, Tensor) else b for b in batch]))

class DataLoader:
    def __init__(self, dataset, y=None, batch_size=32, shuffle=True, drop_last=False, collate_fn=None):
        if y is not None:
            # Direct NumPy array fast-path: zero-overhead slicing
            self.x, self.y = np.asarray(dataset), np.asarray(y)
            self.dataset = None
            self.length = len(self.x)
        else:
            self.dataset = dataset
            self.x = getattr(dataset, "x", None)
            self.y = getattr(dataset, "y", None)
            self.length = len(dataset)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate
        self.indices = np.arange(self.length)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        n = self.length
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n:
                if self.drop_last:
                    break
                end = n
            if self.dataset is None and self.collate_fn is default_collate:
                sl = self.indices[start:end] if self.shuffle else slice(start, end)
                yield Tensor(self.x[sl]), Tensor(self.y[sl])
            else:
                idx = self.indices[start:end]
                batch = [self.dataset[i] if self.dataset is not None else (self.x[i], self.y[i]) for i in idx]
                yield self.collate_fn(batch)

    def __len__(self):
        return self.length // self.batch_size if self.drop_last else (self.length + self.batch_size - 1) // self.batch_size

def _fetch(url, file_path=None):
    # Download remote resource with standard User-Agent header and optional disk caching
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    if file_path is not None:
        with open(file_path, "wb") as f:
            f.write(raw)
    return raw

def _split_tabular(X, y, test_split, normalize):
    # Shuffle, z-score normalize, and split tabular arrays into (train, test) tuples
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    if normalize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    split = int(len(X) * (1 - test_split))
    return (X[:split], y[:split]), (X[split:], y[split:])

def load_mnist(data_dir="data", normalize=True, channels_first=True):
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "mnist.npz")
    if not os.path.exists(file_path):
        _fetch("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", file_path)

    with np.load(file_path) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if channels_first:
        x_train = x_train[:, None, :, :]  # (N, 1, 28, 28)
        x_test = x_test[:, None, :, :]    # (N, 1, 28, 28)

    return (x_train, y_train.astype(np.int64)), (x_test, y_test.astype(np.int64))

def load_wine(data_dir="data", test_split=0.2, normalize=True):
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "winequality.npz")
    if not os.path.exists(file_path):
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        ]
        all_data = []
        for url in urls:
            lines = _fetch(url).decode("utf-8").strip().split("\n")
            all_data.extend([float(x) for x in line.strip().split(";")] for line in lines[1:] if line.strip())
        arr = np.array(all_data, dtype=np.float32)
        X, raw_y = arr[:, :-1], arr[:, -1].astype(np.int64)
        _, y = np.unique(raw_y, return_inverse=True)
        np.savez_compressed(file_path, X=X, y=y)

    with np.load(file_path) as data:
        X, y = data["X"], data["y"]

    return _split_tabular(X, y, test_split, normalize)

def load_iris(data_dir="data", test_split=0.2, normalize=True):
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "iris.data")
    if not os.path.exists(file_path):
        _fetch("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    data, labels = [], []
    label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    for line in lines:
        parts = line.split(",")
        if len(parts) == 5:
            data.append([float(x) for x in parts[:4]])
            labels.append(label_map[parts[4]])

    return _split_tabular(np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64), test_split, normalize)

def load_sine_wave(num_samples=5000, seq_len=20, test_split=0.2):
    series = np.sin(np.linspace(0, 100 * np.pi, num_samples + seq_len, dtype=np.float32))
    X = sliding_window_view(series[:-1], seq_len)[..., None].copy()
    Y = series[seq_len:, None]
    split = int(num_samples * (1 - test_split))
    return (X[:split], Y[:split]), (X[split:], Y[split:])

def load_tiny_shakespeare(data_dir="data", seq_len=32, test_split=0.1, text=None):
    if text is None:
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, "tinyshakespeare.txt")
        if not os.path.exists(file_path):
            _fetch("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    chars = sorted(list(set(text)))
    char2idx = {c: i for i, c in enumerate(chars)}
    encoded = np.array([char2idx[c] for c in text], dtype=np.int64)
    n = (len(encoded) - 1) // seq_len * seq_len
    X = encoded[:n].reshape(-1, seq_len)
    Y = encoded[1 : n + 1].reshape(-1, seq_len)
    split = int(len(X) * (1 - test_split))
    return (X[:split], Y[:split]), (X[split:], Y[split:]), len(chars)

if __name__ == "__main__":
    # Test 1: Sine wave dataset (5000 samples)
    (x_tr_s, y_tr_s), (x_te_s, y_te_s) = load_sine_wave(num_samples=5000, seq_len=15)
    assert x_tr_s.shape == (4000, 15, 1) and y_tr_s.shape == (4000, 1)
    print("Sine wave dataset (5,000 samples) verified!")

    # Test 2: Wine Quality dataset (6,497 samples)
    (x_tr_w, y_tr_w), (x_te_w, y_te_w) = load_wine()
    assert x_tr_w.shape[1] == 11 and len(x_tr_w) + len(x_te_w) == 6497
    print("Wine Quality dataset (6,497 samples) verified!")

    # Test 3: Iris dataset
    (x_train_i, y_train_i), (x_test_i, y_test_i) = load_iris()
    assert x_train_i.shape == (120, 4) and y_train_i.shape == (120,)
    print("Iris dataset verified!")

    # Test 4: MNIST dataset (60,000 samples)
    (x_train_m, y_train_m), (x_test_m, y_test_m) = load_mnist()
    assert x_train_m.shape == (60000, 1, 28, 28)
    print("MNIST dataset (60,000 samples) verified!")

    # Test 5: Dataset & DataLoader with collate_fn and drop_last
    class MockDataset(Dataset):
        def __len__(self): return 10
        def __getitem__(self, idx): return np.array([idx, idx * 2]), idx

    ds = MockDataset()
    loader = DataLoader(ds, batch_size=4, drop_last=True)
    assert len(loader) == 2
    for b_x, b_y in loader:
        assert b_x.shape == (4, 2) and b_y.shape == (4,)
    print("Dataset & DataLoader (collate_fn + drop_last) verified!")

    # Test 6: Tiny Shakespeare dataset
    (x_tr_sh, y_tr_sh), (x_te_sh, y_te_sh), vocab_size = load_tiny_shakespeare(text="First Citizen:\nBefore we proceed any further, hear me speak.\nAll:\nSpeak, speak.\n", seq_len=8)
    assert x_tr_sh.shape[1] == 8 and y_tr_sh.shape[1] == 8 and vocab_size > 0
    print("Tiny Shakespeare dataset verified!")
