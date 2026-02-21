import argparse
import csv
import ctypes
import gzip
import os
import struct
import urllib.request
from datetime import datetime


def _bootstrap_cuda_libs():
    """Ensure Jetson CUDA aux libs (e.g., cuSPARSELt) are discoverable before importing torch."""
    candidates = [
        os.path.expanduser("~/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib"),
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12.6/targets/aarch64-linux/lib",
    ]
    found = [p for p in candidates if os.path.isdir(p)]
    if found:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in current.split(":") if p]
        for p in reversed(found):
            if p not in parts:
                parts.insert(0, p)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

    for p in found:
        lib = os.path.join(p, "libcusparseLt.so.0")
        if os.path.exists(lib):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break


_bootstrap_cuda_libs()

import matplotlib.pyplot as plt
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
        hidden_dim: int = 256,
        num_tables: int = 8,
        num_comparisons: int = 6,
        num_blocks: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(28 * 28, hidden_dim), nn.ReLU()]

        for _ in range(num_blocks):
            layers.append(
                LUTBlock(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_tables=num_tables,
                    num_comparisons=num_comparisons,
                    residual=True,
                )
            )
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 10))
        self.net = nn.Sequential(*layers)

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
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adagrad", "rmsprop"])
    p.add_argument("--train-limit", type=int, default=20000)
    p.add_argument("--test-limit", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-tables", type=int, default=8)
    p.add_argument("--num-comparisons", type=int, default=6)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--figures-dir", type=str, default="figures")
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
        num_tables=args.num_tables,
        num_comparisons=args.num_comparisons,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)

    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adagrad":
        opt = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, alpha=0.99)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running_loss += loss.item() * y.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        lr_now = scheduler.get_last_lr()[0]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_acc": acc,
            "best_acc": best_acc,
            "lr": lr_now,
        })
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} test_acc={acc:.4f} "
            f"best_acc={best_acc:.4f} lr={lr_now:.6f}"
        )

    print(f"FINAL_BEST_ACC={best_acc:.4f}")

    os.makedirs(args.figures_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name if args.run_name else f"mnist_lut_{stamp}"

    csv_path = os.path.join(args.figures_dir, f"{run_name}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_acc", "best_acc", "lr"])
        w.writeheader()
        w.writerows(history)

    epochs = [h["epoch"] for h in history]
    train_loss_vals = [h["train_loss"] for h in history]
    test_acc_vals = [h["test_acc"] for h in history]
    lr_vals = [h["lr"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, train_loss_vals, marker="o")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, test_acc_vals, marker="o")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    axes[2].plot(epochs, lr_vals, marker="o")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")

    fig.suptitle(f"{run_name} (best_acc={best_acc:.4f})")
    fig.tight_layout()
    plot_path = os.path.join(args.figures_dir, f"{run_name}_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"METRICS_CSV={csv_path}")
    print(f"CURVES_PNG={plot_path}")


if __name__ == "__main__":
    main()
