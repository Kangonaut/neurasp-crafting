from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


@dataclass
class Sample:
    class_idx: int
    path: Path


class ItemDataset(Dataset):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.samples = self.__load_samples()
        self.mean, self.std = self.__calc_dataset_stats()

    def __calc_dataset_stats(self):
        tensors = []
        for sample in self.samples:
            img = Image.open(sample.path)
            t = pil_to_tensor(img)
            tensors.append(t)
        data = torch.stack(tensors, dim=1).float()
        return data.mean(dim=(1, 2, 3)), data.std(dim=(1, 2, 3))

    def __load_samples(self):
        samples: list[Sample] = []
        for class_idx, item_dir in enumerate(self.path.iterdir()):
            for img_path in item_dir.iterdir():
                samples.append(
                    Sample(
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
        t = pil_to_tensor(img)
        return (t, sample.class_idx)
