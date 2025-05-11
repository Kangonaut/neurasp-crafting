import itertools
import random
from pathlib import Path

import torch
from clingo import Control
from PIL import Image, ImageFilter
from pydantic_yaml import parse_yaml_file_as
from torch import Tensor, device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Action, Item, StripsConfig
from neurasp.neurasp import NeurASP


def train(
    model: Module,
    dl: DataLoader,
    loss_fn: Module,
    optim: Optimizer,
    device: device,
) -> tuple[float, float]:
    model.train()
    total_loss: float = 0
    num_total: int = 0
    num_correct: int = 0

    for X, y in (progress := tqdm(dl)):
        model.zero_grad()

        X, y = X.to(device), y.to(device)

        pred = model(X)

        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()

        num_total += X.size(0)
        num_correct += (y == torch.argmax(pred, dim=1)).int().sum().item()
        total_loss += loss.item()
        progress.desc = f"[TRAIN] loss: {loss / X.size(0):.4f}"

    return total_loss / num_total, num_correct / num_total


def test(
    model: Module,
    dl: DataLoader,
    loss_fn: Module,
    device: device | str,
    valid: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss: float = 0.0
    num_total: int = 0
    num_correct: int = 0

    with torch.no_grad():
        for X, y in (progress := tqdm(dl)):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            num_total += X.size(0)
            num_correct += (y == torch.argmax(pred, dim=1)).int().sum().item()
            total_loss += loss.item()
            progress.desc = (
                f"[{'VALID' if valid else 'TEST'}]: loss: {loss / X.size(0):.4f}"
            )

    return total_loss / num_total, num_correct / num_total


def get_dataset_samples(ds: Dataset) -> tuple[list[dict[str, Tensor]], list[str]]:
    samples = [sample for sample in ds]
    data = []
    obs = []
    for d, o in samples:
        data.append(d)
        obs.append(o)
    return data, obs


def neurasp_save_model(model: NeurASP, storage_dir: Path, is_best: bool):
    for name, network in model.nnMapping.items():
        model_dir = storage_dir / name
        model_dir.mkdir(exist_ok=True, parents=True)
        torch.save(
            network.state_dict(),
            storage_dir / name / ("best.pt" if is_best else "last.pt"),
        )


def neurasp_load_network_weigths(model: NeurASP, path: Path, best: bool = True) -> None:
    for network_dir in path.iterdir():
        weights_path = network_dir / ("best.pt" if best else "last.pt")
        model.nnMapping[network_dir.name].load_state_dict(
            torch.load(weights_path, weights_only=True)
        )


def neurasp_test(
    model: NeurASP,
    ds: Dataset,
):
    data, obs = get_dataset_samples(ds)
    acc = model.testInferenceResults(data, obs)

    return acc


def neurasp_train_epochs(
    model: NeurASP,
    train_ds: Dataset,
    valid_ds: Dataset,
    num_epochs: int,
    storage_dir: Path | None = None,
):
    if storage_dir:
        storage_dir.mkdir(exist_ok=True, parents=True)

    best_acc: float = 0

    train_data, train_obs = get_dataset_samples(train_ds)
    valid_data, valid_obs = get_dataset_samples(valid_ds)

    valid_acc = model.testInferenceResults(valid_data, valid_obs)
    print(f"validation accuracy before training: {valid_acc:.2f} %")

    for epoch_idx in range(num_epochs):
        print(f"epoch {epoch_idx + 1}/{num_epochs}:")

        model.learn(
            dataList=train_data,
            obsList=train_obs,
            epoch=1,
            batchSize=1,
            bar=True,
            smPickle=None,
        )

        train_acc = model.testInferenceResults(train_data, train_obs)
        valid_acc = model.testInferenceResults(valid_data, valid_obs)

        print(f"\ttraining accuracy: {train_acc:.2f} %")
        print(f"\tvalidation accuracy: {valid_acc:.2f} %")

        if storage_dir and valid_acc > best_acc:
            best_acc = valid_acc
            neurasp_save_model(model, storage_dir, is_best=True)

    if storage_dir:
        neurasp_save_model(model, storage_dir, is_best=False)


def train_epochs(
    model: Module,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    loss_fn: Module,
    optim: Optimizer,
    device: device,
    num_epochs: int,
    storage_dir: Path | None = None,
) -> None:
    best_acc: float = 0

    if storage_dir:
        storage_dir.mkdir(exist_ok=True, parents=True)

    for epoch_idx in range(num_epochs):
        train_loss, train_acc = train(model, train_dl, loss_fn, optim, device)
        print(f"[TRAIN] epoch {epoch_idx + 1}:")
        print(f"\tloss: {train_loss:.4f}")
        print(f"\taccuracy: {train_acc * 100:.2f} %")

        test_loss, test_acc = test(model, valid_dl, loss_fn, device, valid=True)
        print(f"[VALID] epoch {epoch_idx + 1}:")
        print(f"\tloss: {test_loss:.4f}")
        print(f"\taccuracy: {test_acc * 100:.2f} %")

        if storage_dir and test_acc > best_acc:
            best_acc = test_acc
            path = storage_dir / "best.pt"
            torch.save(model.state_dict(), path)
            print(f"successfully saved model to: {path}")

    if storage_dir:
        path = storage_dir / "last.pt"
        torch.save(model.state_dict(), path)
        print(f"successfully saved model to: {path}")


def generate_asp_program(
    actions: list[Action],
    items: list[Item],
    time_steps: int,
    inventory_size: int,
    dataset_generation: bool = False,
    blank_id: int = 0,
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
        # add inventory images
        for idx in range(inventory_size):
            program.append(f"init_img(init_img_{idx}).")
            program.append(f"final_img(final_img_{idx}).")

        # add identify mappings
        program.append("init(X) :- identify(0,I,X), init_img(I).")
        program.append("final(X) :- identify(0,I,X), final_img(I).")

        # add final inventory constraints
        program.append(f":- final(X), not have(X,{time_steps}).")
        program.append(f":- have(X,{time_steps}), not final(X).")

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


def get_run_number(path: Path) -> int:
    prev_runs = map(int, [dir.stem.split("-")[-1] for dir in path.iterdir()])
    return max(itertools.chain(prev_runs, (-1,))) + 1


def generate_neural_atoms(items: list[Item]) -> str:
    item_ids = {item.id for item in items}
    item_ids.add(0)  # blank item

    item_str = ",".join(map(str, item_ids))

    atoms = [
        f"nn(identify(1,I), [{item_str}]) :- init_img(I).",
        f"nn(identify(1,I), [{item_str}]) :- final_img(I).",
    ]
    return "\n".join(atoms)


def generate_samples(
    asp_program: str,
    items: list[Item],
    num_samples: int,
    path: Path,
    time_steps: int,
    inventory_size: int,
    min_init: int = 1,
    num_finals: int = 5,
    img_size: int = 32,
    fillcolor: tuple[int, int, int] = (255, 255, 255),
    img_mode="RGB",
    blank_id: int = 0,
) -> None:
    try:
        path.mkdir(parents=True)
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
