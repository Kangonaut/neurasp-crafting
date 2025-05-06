from pathlib import Path

from utils import generate_asp_program, generate_samples, load_config

TRAIN_SAMPLES: int = 500
VALID_SAMPLES: int = 100
TEST_SAMPLES: int = 100

DS_PATH = Path.cwd() / "dataset"
CONFIG_PATH = Path.cwd() / "strips.yml"

config = load_config(CONFIG_PATH)
asp_program = generate_asp_program(
    actions=config.actions,
    items=config.items,
    time_steps=config.time_steps,
    inventory_size=config.inventory_size,
    dataset_generation=True,
)
for num_samples, split in zip(
    [TRAIN_SAMPLES, VALID_SAMPLES, TEST_SAMPLES], ["train", "valid", "test"]
):
    generate_samples(
        asp_program=asp_program,
        items=config.items,
        num_samples=num_samples,
        time_steps=config.time_steps,
        inventory_size=config.inventory_size,
        path=DS_PATH / split,
    )
