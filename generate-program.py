from argparse import ArgumentParser
from pathlib import Path

from utils import generate_asp_program, load_config

CONFIG_PATH = Path.cwd() / "strips.yml"
config = load_config(CONFIG_PATH)

item_mapping = {item.name: item.id for item in config.items}

parser = ArgumentParser()
parser.add_argument("-i", "--init", action="extend", nargs="+")
args = parser.parse_args()

init = "\n".join(f"init({item_mapping[token]})." for token in (args.init or []))

program = generate_asp_program(
    actions=config.actions,
    items=config.items,
    time_steps=config.time_steps,
    inventory_size=config.inventory_size,
    dataset_generation=True,
)
print(program + "\n" + init)
