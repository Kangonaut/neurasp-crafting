from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from datasets import ItemDataset
from network import ItemClassifier
from utils import load_config, test

CONFIG_PATH = Path.cwd() / "strips.yml"
TEST_PATH = Path.cwd() / "dataset" / "test"

config = load_config(CONFIG_PATH)

# argument parsing
parser = ArgumentParser()
parser.add_argument("-m", "--model", action="store")
args = parser.parse_args()
model_path = Path(args.model)

# dataset
test_ds = ItemDataset(TEST_PATH)
test_dl = DataLoader(test_ds, batch_size=4, shuffle=False)

# neural network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = ItemClassifier(
    num_classes=len(config.items) + 1,  # all items including blank
    neurasp=False,
)
loss_fn = CrossEntropyLoss()

# load network weights
network.load_state_dict(torch.load(model_path, weights_only=True))
network = network.to(device)

# testing
test_loss, test_acc = test(
    model=network,
    dl=test_dl,
    device=device,
    loss_fn=loss_fn,
)
print(f"test loss: {test_loss:.4f}")
print(f"test accuracy: {100 * test_acc:.2f} %")
