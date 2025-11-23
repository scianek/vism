import hashlib
import numpy as np

from pathlib import Path
from typing import Optional
from .types import ImageEmbedding


def _get_cache_dir(model_name: str) -> Path:
    model_name = model_name.replace("/", "_").replace("\\", "_")
    cache_dir = Path.home() / ".cache" / "vism" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _compute_cache_key(path: Path) -> str:
    stat = path.stat()
    key_string = f"{path.absolute()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(key_string.encode()).hexdigest()


def cache_embedding(embedding: ImageEmbedding, model_name: str) -> None:
    cache_dir = _get_cache_dir(model_name)
    cache_key = _compute_cache_key(embedding.path)
    cache_file = cache_dir / f"{cache_key}.npy"

    try:
        np.save(cache_file, embedding.embedding)
    except Exception as e:
        print(f"Warning: Failed to cache embedding for {embedding.path}: {e}")


def load_cached_embedding(path: Path, model_name: str) -> Optional[ImageEmbedding]:
    if not path.exists():
        return None

    cache_dir = _get_cache_dir(model_name)
    cache_key = _compute_cache_key(path)
    cache_file = cache_dir / f"{cache_key}.npy"

    if not cache_file.exists():
        return None

    try:
        embedding_array = np.load(cache_file)
        return ImageEmbedding(path=path, embedding=embedding_array)
    except Exception:
        return None
