from pathlib import Path

import torch
from torch import device
from torch._prims_common import number_type
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: Module,
    dl: DataLoader,
    loss_fn: Module,
    optim: Optimizer,
    device: device,
) -> tuple[float, float]:
    model.train()
    total_loss: float = 0
    num_correct: int = 0

    for X, y in (progress := tqdm(dl)):
        model.zero_grad()

        X, y = X.to(device), y.to(device)

        pred = model(X)

        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()

        num_correct += (y == torch.argmax(pred, dim=1)).int().sum().item()
        total_loss += loss.item()
        progress.desc = f"[TRAIN] loss: {loss / X.size(0):.4f}"

    num_samples = len(dl) * X.size(0)  # type: ignore
    return total_loss / num_samples, num_correct / num_samples


def test(
    model: Module,
    dl: DataLoader,
    loss_fn: Module,
    device: device | str,
) -> tuple[float, float]:
    model.eval()
    total_loss: float = 0.0
    num_correct: int = 0

    with torch.no_grad():
        for X, y in (progress := tqdm(dl)):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            num_correct += (y == torch.argmax(pred, dim=1)).int().sum().item()
            total_loss += loss.item()
            progress.desc = f"[TEST]: loss: {loss / X.size(0):.4f}"

    num_samples = len(dl) * X.size(0)  # type: ignore
    return total_loss / num_samples, num_correct / num_samples


def train_epochs(
    model: Module,
    train_dl: DataLoader,
    test_dl: DataLoader,
    loss_fn: Module,
    optim: Optimizer,
    device: device,
    num_epochs: int,
    storage_dir: Path | None = None,
) -> None:
    best_acc: float = 0

    if storage_dir:
        storage_dir.mkdir(exist_ok=True, parents=True)

    for epoch_idx in range(num_epochs):
        train_loss, train_acc = train(model, train_dl, loss_fn, optim, device)
        print(f"[TRAIN] epoch {epoch_idx + 1}:")
        print(f"\tloss: {train_loss:.4f}")
        print(f"\taccuracy: {train_acc:.4f}")

        test_loss, test_acc = test(model, test_dl, loss_fn, device)
        print(f"[TEST] epoch {epoch_idx + 1}:")
        print(f"\tloss: {test_loss:.4f}")
        print(f"\taccuracy: {test_acc:.4f}")

        if storage_dir and test_acc > best_acc:
            torch.save(model, storage_dir / "best.pt")

    if storage_dir:
        torch.save(model, storage_dir / "last.pt")
