import itertools
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class ItemDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.img_paths = self.__get_img_paths()
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
        return list(self.path.glob("./*/*.png"))

    def __calc_img_stats(self):
        tensors = []
        transform = transforms.Compose([transforms.ToTensor()])
        for img_path in self.__get_img_paths():
            img = Image.open(img_path)
            t = transform(img)
            tensors.append(t)
        data = torch.stack(tensors, dim=1).float()
        return data.mean(dim=(1, 2, 3)), data.std(dim=(1, 2, 3))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> tuple[Tensor, int]:
        img_path = self.img_paths[idx]
        x: Tensor = self.transform(Image.open(img_path))  # type: ignore
        class_id = int(img_path.stem.split("_")[-1])

        y = -1
        with open(img_path.parent / "label.txt", "r") as file:
            lines = file.read().splitlines()
            init, final = [list(map(int, line.split(" "))) for line in lines]

            if img_path.stem.startswith("init"):
                y = init[class_id]
            else:
                y = final[class_id]

        return x, y


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
        transform = transforms.Compose([transforms.ToTensor()])
        for img_path in self.__get_img_paths():
            img = Image.open(img_path)
            t = transform(img)
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
            path.stem: torch.unsqueeze(
                self.transform(Image.open(path)),  # type: ignore
                dim=0,
            )
            for path in itertools.chain(init_paths, final_paths)
        }  # type: ignore

        with open(sample_path / "label.txt", "r") as file:
            lines = file.read().splitlines()
            init, final = [list(map(int, line.split(" "))) for line in lines]

            obs = "\n".join(
                itertools.chain(
                    map(lambda x: f":- not init({x}).", init),
                    map(lambda x: f":- not final({x}).", final),
                )
            )

        return imgs, obs
