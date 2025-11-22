from pathlib import Path
from PIL import Image
from typing import List
from .types import ImageData


def find_images_recursive(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tiff")
    paths = []
    for ext in exts:
        paths.extend(directory.rglob(ext))
        paths.extend(directory.rglob(ext.upper()))
    return sorted(set(paths))


def load_image(path: Path) -> ImageData:
    with open(path, "rb") as f:
        return ImageData(path=path, image=Image.open(f).convert("RGB"))
