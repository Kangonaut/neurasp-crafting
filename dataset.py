import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import normalize, pil_to_tensor


class InventoryDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.sample_paths = list(self.path.iterdir())
        self.mean, self.std = self.__calc_img_stats()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

    def __get_img_paths(self) -> list[Path]:
        paths: list[Path] = []
        for sample_path in self.path.iterdir():
            paths.extend(list(sample_path.glob("*.png")))
        return paths

    def __calc_img_stats(self):
        tensors = []
        for img_path in self.__get_img_paths():
            img = Image.open(img_path)
            t = pil_to_tensor(img)
            tensors.append(t)
        data = torch.stack(tensors, dim=1).float()
        return data.mean(dim=(1, 2, 3)), data.std(dim=(1, 2, 3))

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx) -> tuple[dict[str, Tensor], str]:
        sample_path = self.sample_paths[idx]

        init_paths = sample_path.glob("init_img_*.png")
        final_paths = sample_path.glob("final_img_*.png")

        imgs: dict[str, Tensor] = {
            path.stem: torch.unsqueeze(self.transform(Image.open(path)), dim=0)  # type: ignore
            for path in itertools.chain(init_paths, final_paths)
        }  # type: ignore

        with open(sample_path / "label.txt", "r") as file:
            lines = file.read().splitlines()
            init, final = [line.split(" ") for line in lines]

            obs = "\n".join(
                itertools.chain(
                    map(lambda x: f":- not init({x}).", init),
                    map(lambda x: f":- not final({x}).", final),
                )
            )

        return imgs, obs


class Test2Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.axe_path = Path.cwd() / "data" / "axe"
        self.wood_path = Path.cwd() / "data" / "wood"

        self.coffee_beans_path = Path.cwd() / "data" / "coffee-beans"
        self.coffee_path = Path.cwd() / "data" / "coffee"

        self.samples = []
        for idx in range(50):
            axe = self.axe_path / f"{idx}.png"
            self.samples.append((axe, 0))

            wood = self.wood_path / f"{idx}.png"
            self.samples.append((wood, 1))

            coffee_beans = self.coffee_beans_path / f"{idx}.png"
            self.samples.append((coffee_beans, 2))

            coffee = self.coffee_path / f"{idx}.png"
            self.samples.append((coffee, 3))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = (pil_to_tensor(Image.open(sample[0])).float() - 128) / 128
        return img, sample[1]


class TestDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.axe_path = Path.cwd() / "data" / "axe"
        self.wood_path = Path.cwd() / "data" / "wood"

        self.coffee_beans_path = Path.cwd() / "data" / "coffee-beans"
        self.coffee_path = Path.cwd() / "data" / "coffee"

        self.samples = []
        for idx in range(50):
            axe = self.axe_path / f"{idx}.png"
            wood = self.wood_path / f"{idx}.png"

            self.samples.append((axe, wood, (0, 1)))

            coffee_beans = self.coffee_beans_path / f"{idx}.png"
            coffee = self.coffee_path / f"{idx}.png"

            self.samples.append((coffee_beans, coffee, (2, 3)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_1 = (pil_to_tensor(Image.open(sample[0])).float() - 128) / 128
        img_2 = (pil_to_tensor(Image.open(sample[1])).float() - 128) / 128
        pre = sample[2][0]
        post = sample[2][1]

        return img_1, img_2, pre, post


@dataclass
class ItemSample:
    class_idx: int
    path: Path


class ItemDataset(Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.num_classes = len(list(self.path.iterdir()))
        self.samples = self.__load_samples()
        self.mean, self.std = map(Tensor.tolist, self.__calc_dataset_stats())

    def __calc_dataset_stats(self):
        tensors = []
        for sample in self.samples:
            img = Image.open(sample.path)
            t = pil_to_tensor(img)
            tensors.append(t)
        data = torch.stack(tensors, dim=1).float()
        return data.mean(dim=(1, 2, 3)), data.std(dim=(1, 2, 3))

    def __load_samples(self):
        samples: list[ItemSample] = []
        for class_idx, item_dir in enumerate(self.path.iterdir()):
            for img_path in item_dir.iterdir():
                samples.append(
                    ItemSample(
                        class_idx=class_idx,
                        path=img_path,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Tensor, int]:
        sample = self.samples[idx]
        img = Image.open(sample.path)
        t = pil_to_tensor(img).float()
        t = normalize(t, self.mean, self.std)
        return (t, sample.class_idx)
