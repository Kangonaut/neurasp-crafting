from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import ItemDataset
from network import ItemClassifier
from utils import get_run_number, load_config, test, train_epochs

CONFIG_PATH = Path.cwd() / "strips.yml"
TRAIN_PATH = Path.cwd() / "dataset" / "train"
VALID_PATH = Path.cwd() / "dataset" / "valid"
TEST_PATH = Path.cwd() / "dataset" / "test"
STORAGE_DIR = Path.cwd() / "results" / "network"

STORAGE_DIR.mkdir(exist_ok=True, parents=True)
storage_dir = STORAGE_DIR / f"train-{get_run_number(STORAGE_DIR)}"

config = load_config(CONFIG_PATH)

# dataset
train_ds = ItemDataset(TRAIN_PATH)
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
valid_ds = ItemDataset(VALID_PATH)
valid_dl = DataLoader(valid_ds, batch_size=4, shuffle=False)
test_ds = ItemDataset(TEST_PATH)
test_dl = DataLoader(test_ds, batch_size=4, shuffle=False)

# neural network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = ItemClassifier(
    num_classes=len(config.items) + 1,  # all items including blank
    neurasp=False,
).to(device)
optim = Adam(network.parameters(), lr=0.0001)
loss_fn = CrossEntropyLoss()

# training
train_epochs(
    model=network,
    train_dl=train_dl,
    valid_dl=valid_dl,
    loss_fn=loss_fn,
    optim=optim,
    device=device,
    num_epochs=5,
    storage_dir=storage_dir,
)

# testing
test_loss, test_acc = test(
    model=network,
    dl=test_dl,
    device=device,
    loss_fn=loss_fn,
)
print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {100 * test_acc:.2f} %")
