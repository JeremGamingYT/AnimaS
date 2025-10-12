from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def default_transforms(image_size: int, is_train: bool) -> Callable[[Image.Image], torch.Tensor]:
    if is_train:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )


class FramePairDataset(Dataset):
    """Yields consecutive frame pairs (x_t, x_{t+1}) from a folder of frames or nested subfolders.

    Directory structures supported:
      - root/ contains sorted frames -> pairs are (frame[i], frame[i+1])
      - root/sequence_*/ contains sorted frames per subfolder -> pairs across each subfolder
    """

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        is_train: bool = True,
        sequence_by_subfolder: bool = False,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.sequence_by_subfolder = bool(sequence_by_subfolder)
        self.transform = transform or default_transforms(self.image_size, is_train)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.pairs: List[Tuple[Path, Path]] = []
        if self.sequence_by_subfolder:
            for sub in sorted([p for p in self.root.iterdir() if p.is_dir()]):
                frames = sorted([p for p in sub.iterdir() if p.is_file() and _is_image(p)])
                for i in range(len(frames) - 1):
                    self.pairs.append((frames[i], frames[i + 1]))
        else:
            frames = sorted([p for p in self.root.iterdir() if p.is_file() and _is_image(p)])
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i], frames[i + 1]))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No consecutive pairs found in: {self.root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path_t, path_tp1 = self.pairs[index]
        img_t = Image.open(path_t).convert("RGB")
        img_tp1 = Image.open(path_tp1).convert("RGB")
        x_t = self.transform(img_t)
        x_tp1 = self.transform(img_tp1)
        return x_t, x_tp1


