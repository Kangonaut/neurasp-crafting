import random
from pathlib import Path

from clingo import Control
from PIL import Image, ImageFilter
from pydantic_yaml import parse_yaml_file_as

from config import Action, Item, StripsConfig


def add_margin(
    img: Image.Image,
    pos: tuple[int, int],
    inner_size: int,
    outer_size: int,
    fillcolor: tuple[int, int, int],
    mode="RGB",
) -> Image.Image:
    x, y = pos
    res = Image.new(mode, (outer_size, outer_size), fillcolor)
    resized_img = img.resize((inner_size, inner_size))
    res.paste(resized_img, (x, y))
    return res


def augment_image(
    img: Image.Image,
    img_size: int,
    fillcolor: tuple[int, int, int],
) -> Image.Image:
    angle = random.random() * 180 - 90
    img = img.rotate(angle, expand=True, fillcolor=fillcolor)

    size = random.randint(img_size, img_size)
    pos = tuple(random.randint(0, img_size - size) for _ in range(2))
    img = add_margin(img, pos, inner_size=size, outer_size=img_size, fillcolor=fillcolor)  # type: ignore

    delta = size // 5  # how much of the image we allow to be obscured
    delta_x = random.randint(-(img_size - (pos[0] + size) + delta), pos[0] + delta)
    delta_y = random.randint(-(img_size - (pos[1] + size) + delta), pos[1] + delta)
    img = img.transform(
        size=img.size,
        method=Image.AFFINE,  # type: ignore
        data=(1, 0, delta_x, 0, 1, delta_y),
        fillcolor=fillcolor,
    )

    radius = random.randint(1, 3)
    img = img.filter(ImageFilter.GaussianBlur(radius))

    return img


def load_config(path: Path) -> StripsConfig:
    with open(path, "r") as file:
        return parse_yaml_file_as(StripsConfig, file)


def generate_samples(
    asp_program: str,
    items: list[Item],
    num_samples: int,
    path: Path,
    time_steps: int,
    min_init: int = 3,
    max_init: int = 8,
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
        num_init = random.randint(min_init, max_init)
        init = random.sample(item_ids, k=num_init)
        init.extend(
            [blank_id] * (max_init - len(init))
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
        finals = [f + (blank_id,) * (max_init - len(f)) for f in finals]

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
                img.save(sample_path / f"init-{idx}.png")

            # generate final images
            for idx, item in enumerate(final):
                img = item_imgs[item]
                img = augment_image(img, img_size, fillcolor)
                img.save(sample_path / f"final-{idx}.png")

        curr_samples += len(finals)


def generate_asp_program(
    actions: list[Action],
    items: list[Item],
    time_steps: int,
    dataset_generation: bool,
    blank_id: int = -1,
) -> str:
    item_mapping: dict[str, int] = {item.name: item.id for item in items}

    program: list[str] = []

    # fundamental rules
    program = [
        f"time(1..{time_steps}).",
        # initial inventory
        f"have(X,0) :- init(X).",
        # action rules
        f"{{ occur(A,T) : action(A) }} = 1 :- time(T).",
        f":- occur(A,T), pre(A,I), not have(I,T-1).",
        f"have(I,T) :- time(T), have(I,T-1), not occur(A,T) : del(A,I).",
        f"have(I,T) :- occur(A,T), add(A,I).",
    ]

    if not dataset_generation:
        program.append(f":- goal(X), not have(X,{time_steps}).")
        program.append(f":- have(X,{time_steps}), not goal(X).")

    # add items
    program.append(f"item({blank_id}).")
    for item in items:
        program.append(f"item({item.id}).")

    # add actions
    for action in actions:
        program.append(f"action({action.name}).")

        for x in action.preconditions:
            program.append(f"pre({action.name},{item_mapping[x]}).")

        for x in action.add_list:
            program.append(f"add({action.name},{item_mapping[x]}).")

        for x in action.delete_list:
            program.append(f"del({action.name},{item_mapping[x]}).")

    return "\n".join(program)


TIME_STEPS = 3
config = load_config(Path.cwd() / "strips.yml")
asp_program = generate_asp_program(
    actions=config.actions,
    items=config.items,
    time_steps=TIME_STEPS,
    dataset_generation=True,
)
print(asp_program)
generate_samples(
    asp_program=asp_program,
    items=config.items,
    num_samples=10,
    time_steps=TIME_STEPS,
    path=Path.cwd() / "data",
)
