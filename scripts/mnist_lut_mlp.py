import argparse
import gzip
import os
import struct
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lut_nn import LUTBlock


MNIST_URL_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_if_missing(root: str):
    os.makedirs(root, exist_ok=True)
    for fname in MNIST_FILES.values():
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{MNIST_URL_BASE}/{fname}", path)


def _read_idx_images(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad image magic number: {magic}")
        data = f.read()
    x = torch.frombuffer(data, dtype=torch.uint8).clone().float()
    x = x.view(n, rows, cols) / 255.0
    return x


def _read_idx_labels(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad label magic number: {magic}")
        data = f.read()
    y = torch.frombuffer(data, dtype=torch.uint8).clone().long()
    return y


def load_mnist(root: str = "./data/mnist"):
    _download_if_missing(root)
    x_train = _read_idx_images(os.path.join(root, MNIST_FILES["train_images"]))
    y_train = _read_idx_labels(os.path.join(root, MNIST_FILES["train_labels"]))
    x_test = _read_idx_images(os.path.join(root, MNIST_FILES["test_images"]))
    y_test = _read_idx_labels(os.path.join(root, MNIST_FILES["test_labels"]))
    return x_train, y_train, x_test, y_test


class MLPWithLUT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        lut_out: int = 128,
        num_tables: int = 8,
        num_comparisons: int = 6,
        tau: float = 0.5,
        routing_sharpness: float = 12.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            LUTBlock(
                in_features=hidden_dim,
                out_features=lut_out,
                num_tables=num_tables,
                num_comparisons=num_comparisons,
                tau=tau,
                routing_sharpness=routing_sharpness,
                residual=(hidden_dim == lut_out),
            ),
            nn.ReLU(),
            nn.Linear(lut_out, 10),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-limit", type=int, default=10000)
    p.add_argument("--test-limit", type=int, default=2000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lut-out", type=int, default=128)
    p.add_argument("--num-tables", type=int, default=8)
    p.add_argument("--num-comparisons", type=int, default=6)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--routing-sharpness", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device={device}")

    x_train, y_train, x_test, y_test = load_mnist("./data/mnist")

    if args.train_limit > 0:
        x_train = x_train[: args.train_limit]
        y_train = y_train[: args.train_limit]
    if args.test_limit > 0:
        x_test = x_test[: args.test_limit]
        y_test = y_test[: args.test_limit]

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MLPWithLUT(
        hidden_dim=args.hidden,
        lut_out=args.lut_out,
        num_tables=args.num_tables,
        num_comparisons=args.num_comparisons,
        tau=args.tau,
        routing_sharpness=args.routing_sharpness,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
