import shutil
from pathlib import Path

from pydantic_yaml import parse_yaml_file_as

from config import SelectionConfig

INPUT_PATH = Path.cwd() / "sprites" / "original"
OUTPUT_PATH = Path.cwd() / "sprites" / "selected"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path.cwd() / "selection.yml"


with open(CONFIG_PATH, "r") as file:
    config = parse_yaml_file_as(SelectionConfig, file)

for item in config.items:
    shutil.copyfile(INPUT_PATH / f"{item.src}.png", OUTPUT_PATH / f"{item.name}.png")
