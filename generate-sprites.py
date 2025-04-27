import itertools
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_PATH = Path.cwd() / "sprites.png"

OUTPUT_PATH = Path.cwd() / "sprites"
OUTPUT_PATH.mkdir(exist_ok=True)

img = Image.open(INPUT_PATH)
img = np.array(img)

NUM_COLS = 16
NUM_ROWS = 22
IMG_SIZE = 32

for row, col in itertools.product(range(NUM_ROWS), range(NUM_COLS)):
    sprite = img[
        IMG_SIZE * row : IMG_SIZE * (row + 1), IMG_SIZE * col : IMG_SIZE * (col + 1)
    ]

    # skip empty sprites
    if sprite.std() == 0:
        continue

    Image.fromarray(sprite).save(OUTPUT_PATH / f"{row}-{col}.png")
