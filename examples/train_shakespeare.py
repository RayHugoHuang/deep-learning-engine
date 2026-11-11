import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import dle.nn as nn
from dle.tensor import Tensor, no_grad
from dle.optim import AdamW, CosineAnnealingLR
from dle.quant import quantize_dynamic
from dle.utils import summary, train_epoch, validate_model, save_model, load_model
from dataset import DataLoader, load_tiny_shakespeare

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))

    def forward(self, x, mask, rotary_emb):
        x = x + self.attn(self.ln1(x), mask=mask, rotary_emb=rotary_emb)
        return x + self.mlp(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=32, num_heads=16, num_layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.rope = nn.RotaryEmbedding(d_model // num_heads)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        tok = self.emb(x)
        mask = nn.causal_mask(tok.data.shape[1])
        for block in self.blocks:
            tok = block(tok, mask=mask, rotary_emb=self.rope)
        return self.head(self.ln(tok))

    @no_grad()
    def generate(self, idx, max_new_tokens=80):
        self.eval()
        idx = list(idx)
        for _ in range(max_new_tokens):
            inp = Tensor(np.array([idx[-32:]], dtype=np.int64))
            nxt = int(np.argmax(self(inp).data[0, -1]))
            idx.append(nxt)
        return idx

def main():
    print("=== NLP Causal Language Modeling: Tiny Shakespeare MiniGPT ===")
    (x_tr, y_tr), (x_te, y_te), vocab_size = load_tiny_shakespeare(seq_len=32)
    train_loader = DataLoader(x_tr[:5000], y_tr[:5000], batch_size=64)
    test_loader = DataLoader(x_te[:200], y_te[:200], batch_size=64, shuffle=False)

    model = MiniGPT(vocab_size)
    summary(model, train_loader)

    optimizer = AdamW(model.parameters(), lr=4e-3, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=12, eta_min=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(12):
        train_epoch(model, train_loader, loss_fn, optimizer, epoch=epoch, epochs=12, scheduler=scheduler)
    validate_model(model, test_loader, mode="classification")

    # Delivery: Checkpoint persistence, quantization & autoregressive generation
    ckpt_path = "shakespeare_gpt.pth"
    save_model(model, ckpt_path)
    load_model(model, ckpt_path)
    if os.path.exists(ckpt_path): os.remove(ckpt_path)

    quantized = quantize_dynamic(model)
    chars = sorted(list(set(open("data/tinyshakespeare.txt", "r", encoding="utf-8").read())))
    char2idx, idx2char = {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}
    prompt = "ROMEO:"
    gen_ids = quantized.generate([char2idx[c] for c in prompt], max_new_tokens=60)
    print("--- Autoregressive Sample Generation (Quantized Model) ---")
    print("".join(idx2char[i] for i in gen_ids) + "\n")

if __name__ == "__main__":
    main()
