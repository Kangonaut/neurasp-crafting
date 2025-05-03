import random
from os import wait
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from pydantic_yaml import parse_yaml_file_as
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Action, Item, StripsConfig


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
            progress.desc = f"[TEST]: loss: {loss / X.size(0):.4f}"

    return total_loss / num_total, num_correct / num_total


def train_epochs(
    model: Module,
    train_dl: DataLoader,
    test_dl: DataLoader,
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
        print(f"\taccuracy: {train_acc:.4f}")

        test_loss, test_acc = test(model, test_dl, loss_fn, device)
        print(f"[TEST] epoch {epoch_idx + 1}:")
        print(f"\tloss: {test_loss:.4f}")
        print(f"\taccuracy: {test_acc:.4f}")

        if storage_dir and test_acc > best_acc:
            torch.save(model, storage_dir / "best.pt")

    if storage_dir:
        torch.save(model, storage_dir / "last.pt")


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
            program.append(f"final_img(init_img_{idx}).")

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
