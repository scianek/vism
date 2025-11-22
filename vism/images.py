import os
from pathlib import Path
from PIL import Image
from typing import List
from .types import ImageData


def find_images_recursive(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in valid_exts:
                paths.append(Path(root) / file)

    return sorted(paths)


def load_image(path: Path) -> ImageData:
    with open(path, "rb") as f:
        return ImageData(path=path, image=Image.open(f).convert("RGB"))
