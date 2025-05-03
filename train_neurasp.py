from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import InventoryDataset, Test2Dataset, TestDataset
from network import ItemClassifier
from neurasp.neurasp import NeurASP
from utils import generate_asp_program, load_config

config = load_config(path=Path.cwd() / "strips.yml")
asp_program = generate_asp_program(
    actions=config.actions,
    items=config.items,
    time_steps=config.time_steps,
    inventory_size=config.inventory_size,
)

item_ids = {item.id for item in config.items}
neural_atoms = f"""
nn(identify(1,I), [{','.join(map(str, item_ids))}]) :- init_img(I).
nn(identify(1,I), [{','.join(map(str, item_ids))}]) :- final_img(I).
"""

# dataset
train_ds = InventoryDataset(path=Path.cwd() / "data")
samples = [sample for sample in train_ds]
data_list = [sample[0] for sample in samples]
obs_list = [sample[1] for sample in samples]

# neural network
network = ItemClassifier(num_classes=len(config.items))
nn_mapping = {"identify": network}
optimizers = {"identify": torch.optim.Adam(network.parameters(), lr=0.0001)}

# NeurASP model
model = NeurASP(neural_atoms + asp_program, nn_mapping, optimizers)

acc = model.testInferenceResults(data_list, obs_list)
print(f"[PRE] accuracy: {acc}")

for idx in range(10):
    model.learn(dataList=data_list, obsList=obs_list, epoch=1, batchSize=1, bar=True)
    acc = model.testInferenceResults(data_list, obs_list)
    print(f"[{idx}] accuracy: {acc}")

print(model.infer(data_list[1]))
