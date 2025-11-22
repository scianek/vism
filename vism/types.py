from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import numpy as np


@dataclass(slots=True, frozen=True)
class ImageData:
    path: Path
    image: Image.Image


@dataclass(slots=True, frozen=True)
class ImageEmbedding:
    path: Path
    embedding: np.ndarray


@dataclass(slots=True, frozen=True)
class SearchResult:
    path: Path
    score: float
