import random
from pathlib import Path

from clingo import Control
from PIL import Image

from config import Item
from utils import augment_image, generate_asp_program, load_config


def generate_samples(
    asp_program: str,
    items: list[Item],
    num_samples: int,
    path: Path,
    time_steps: int,
    inventory_size: int,
    min_init: int = 3,
    num_finals: int = 5,
    img_size: int = 32,
    fillcolor: tuple[int, int, int] = (255, 255, 255),
    img_mode="RGB",
    blank_id: int = -1,
) -> None:
    try:
        path.mkdir()
    except FileExistsError:
        raise Exception("dataset already exists")

    curr_samples: int = 0

    item_mapping = {item.id: item for item in items}
    item_ids = list(item_mapping.keys())
    item_imgs: dict[int, Image.Image] = {
        item.id: Image.open(item.path) for item in items
    }

    # add blank img
    item_imgs[blank_id] = Image.new(
        mode=img_mode, size=(img_size, img_size), color=fillcolor
    )

    while curr_samples < num_samples:
        # generate initial inventory
        num_init = random.randint(min_init, inventory_size)
        init = random.sample(item_ids, k=num_init)
        init.extend(
            [blank_id] * (inventory_size - len(init))
        )  # fill remaining spaces with blank

        # generate facts for initial inventory
        init_facts = ""
        for item in init:
            init_facts += f"init({item})."

        # generate potential final inventories
        # NOTE: 0 ... find all stable models
        control = Control(["--warn=none", "0"])
        control.add("base", [], asp_program + init_facts)
        control.ground([("base", [])])
        models = []
        control.solve(on_model=lambda model: models.append(model.symbols(atoms=True)))
        finals = set(
            tuple(
                map(
                    lambda s: s.arguments[0].number,
                    filter(
                        lambda s: s.name == "have"
                        and s.arguments[1].number == time_steps,
                        model,
                    ),
                )
            )
            for model in models
        )

        # randomly sample k potential final inventories
        remain_samples = num_samples - curr_samples
        finals = random.sample(
            sorted(finals), k=min(remain_samples, len(finals), num_finals)
        )

        # fill remaining spaces with blank
        finals = [f + (blank_id,) * (inventory_size - len(f)) for f in finals]

        # create the sample
        for sample_idx, final in enumerate(finals, start=curr_samples):
            sample_path = path / str(sample_idx)
            sample_path.mkdir()

            # generate label
            with open(sample_path / "label.txt", "w+") as file:
                content = " ".join(map(str, init)) + "\n" + " ".join(map(str, final))
                file.write(content)

            # generate init images
            for idx, item in enumerate(init):
                img = item_imgs[item]
                img = augment_image(img, img_size, fillcolor)
                img.save(sample_path / f"init_img_{idx}.png")

            # generate final images
            for idx, item in enumerate(final):
                img = item_imgs[item]
                img = augment_image(img, img_size, fillcolor)
                img.save(sample_path / f"final_img_{idx}.png")

        curr_samples += len(finals)


config = load_config(Path.cwd() / "strips.yml")
asp_program = generate_asp_program(
    actions=config.actions,
    items=config.items,
    time_steps=config.time_steps,
    inventory_size=config.inventory_size,
    dataset_generation=True,
)
print(asp_program)
generate_samples(
    asp_program=asp_program,
    items=config.items,
    num_samples=10,
    time_steps=config.time_steps,
    inventory_size=config.inventory_size,
    path=Path.cwd() / "data",
)
