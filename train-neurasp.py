from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import InventoryDataset, ItemDataset
from network import ItemClassifier
from neurasp.neurasp import NeurASP
from utils import (generate_asp_program, generate_neural_atoms, get_run_number,
                   load_config, neurasp_test, neurasp_train_epochs)

NUM_EPOCHS: int = 1
LR: float = 0.0001

CONFIG_PATH = Path.cwd() / "strips.yml"
TRAIN_PATH = Path.cwd() / "dataset" / "train"
VALID_PATH = Path.cwd() / "dataset" / "valid"
TEST_PATH = Path.cwd() / "dataset" / "test"
STORAGE_DIR = Path.cwd() / "results" / "neurasp"

STORAGE_DIR.mkdir(exist_ok=True, parents=True)
storage_dir = STORAGE_DIR / f"train-{get_run_number(STORAGE_DIR)}"

config = load_config(CONFIG_PATH)

# generate program
asp_program = generate_asp_program(
    actions=config.actions,
    items=config.items,
    time_steps=config.time_steps,
    inventory_size=config.inventory_size,
)
neural_atoms = generate_neural_atoms(config.items)

# dataset
train_ds = InventoryDataset(TRAIN_PATH)
valid_ds = InventoryDataset(VALID_PATH)
test_ds = InventoryDataset(TEST_PATH)
item_test_ds = ItemDataset(TEST_PATH)
item_test_dl = DataLoader(item_test_ds, batch_size=4, shuffle=True)

# neural network
network = ItemClassifier(
    num_classes=len(config.items) + 1,  # all items including blank
)
nn_mapping = {"identify": network}
optimizers = {"identify": torch.optim.Adam(network.parameters(), lr=LR)}

# NeurASP model
model = NeurASP(
    dprogram=neural_atoms + "\n" + asp_program,
    nnMapping=nn_mapping,
    optimizers=optimizers,
)

# training
neurasp_train_epochs(
    model=model,
    train_ds=train_ds,
    valid_ds=valid_ds,
    num_epochs=NUM_EPOCHS,
    storage_dir=storage_dir,
)

# testing
neurasp_acc = neurasp_test(
    model=model,
    ds=test_ds,
)
print(f"NeurASP test accuracy: {neurasp_acc:.2f} %")

nn_acc, _ = model.testNN("identify", item_test_dl)
print(f"CNN test accuracy: {neurasp_acc:.2f} %")
