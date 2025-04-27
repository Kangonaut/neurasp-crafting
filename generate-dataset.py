import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

INPUT_PATH = Path.cwd() / "sprites" / "selected"
OUTPUT_PATH = Path.cwd() / "data"
OUTPUT_PATH.mkdir(exist_ok=True)

IMG_MODE = "RGB"
IMG_SIZE = 224
FILLCOLOR = (255, 255, 255)


@dataclass
class Item:
    name: str
    path: Path
    img: Image.Image


def load_items(path: Path) -> list[Item]:
    items = []
    for img_path in path.iterdir():
        items.append(
            Item(
                name=img_path.stem,
                path=img_path,
                img=Image.open(img_path),
            )
        )
    return items


def add_margin(
    img: Image.Image, pos: tuple[int, int], size: int, fillcolor: tuple[int, int, int]
) -> Image.Image:
    x, y = pos
    res = Image.new(IMG_MODE, (IMG_SIZE, IMG_SIZE), fillcolor)
    resized_img = img.resize((size, size))
    res.paste(resized_img, (x, y))
    return res


def generate_samples(item: Item, out_path: Path, num_samples: int) -> None:
    out_path.mkdir(exist_ok=True, parents=True)

    for idx in range(num_samples):
        sample = item.img

        angle = random.random() * 180 - 90
        sample = sample.rotate(angle, expand=True, fillcolor=FILLCOLOR)

        size = random.randint(16, IMG_SIZE)
        pos = tuple(random.randint(0, IMG_SIZE - size) for _ in range(2))
        sample = add_margin(sample, pos, size=size, fillcolor=FILLCOLOR)  # type: ignore

        delta = size // 2  # how much of the image we allow to be obscured
        delta_x = random.randint(-(IMG_SIZE - (pos[0] + size) + delta), pos[0] + delta)
        delta_y = random.randint(-(IMG_SIZE - (pos[1] + size) + delta), pos[1] + delta)
        sample = sample.transform(
            size=sample.size,
            method=Image.AFFINE,  # type: ignore
            data=(1, 0, delta_x, 0, 1, delta_y),
            fillcolor=FILLCOLOR,
        )

        radius = random.randint(1, 10)
        sample = sample.filter(ImageFilter.GaussianBlur(radius))

        sample.save(out_path / f"{item.name}-{idx}.png")


items = load_items(INPUT_PATH)

for item in items:
    generate_samples(item, OUTPUT_PATH / item.name, 50)
