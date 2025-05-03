from itertools import chain
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import ItemDataset
from network import ItemClassifier
from utils import train_epochs

TRAIN_PATH = Path.cwd() / "data"
RESULTS_DIR = Path.cwd() / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def get_run_number() -> int:
    return max(
        chain(
            map(int, [dir.stem.split("-")[-1] for dir in RESULTS_DIR.iterdir()]), (0,)
        ),
    )


run_number = get_run_number()
results_path = RESULTS_DIR / f"train-{run_number}"
results_path.mkdir(exist_ok=True, parents=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_ds = ItemDataset(TRAIN_PATH)
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)

network = ItemClassifier(train_ds.num_classes).to(device)
loss_fn = CrossEntropyLoss()
optim = Adam(network.parameters(), lr=0.0001)


train_epochs(
    network,
    train_dl,
    train_dl,
    loss_fn,
    optim,
    device,
    num_epochs=5,
    storage_dir=results_path / "models",
)
# TODO: train/test split
