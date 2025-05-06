from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader

from datasets import InventoryDataset, ItemDataset
from network import ItemClassifier
from neurasp.neurasp import NeurASP
from utils import (generate_asp_program, generate_neural_atoms, load_config,
                   neurasp_load_network_weigths, neurasp_test)

CONFIG_PATH = Path.cwd() / "strips.yml"
TEST_PATH = Path.cwd() / "dataset" / "test"

# argument parsing
parser = ArgumentParser()
parser.add_argument("-m", "--model", action="store")
args = parser.parse_args()
model_path = Path(args.model)

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
test_ds = InventoryDataset(TEST_PATH)
item_test_ds = ItemDataset(TEST_PATH)
item_test_dl = DataLoader(item_test_ds, batch_size=4, shuffle=True)

# neural network
network = ItemClassifier(
    num_classes=len(config.items) + 1,  # all items including blank
)
nn_mapping = {"identify": network}
optimizers = {}

# NeurASP model
model = NeurASP(
    dprogram=neural_atoms + "\n" + asp_program,
    nnMapping=nn_mapping,
    optimizers=optimizers,
)

# load weights
neurasp_load_network_weigths(model, model_path, best=True)

# testing
neurasp_acc = neurasp_test(model=model, ds=test_ds)
print(f"NeurASP test accuracy: {neurasp_acc:.2f} %")

nn_acc, _ = model.testNN("identify", item_test_dl)
print(f"CNN test accuracy: {neurasp_acc:.2f} %")
