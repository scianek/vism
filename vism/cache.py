import sqlite3
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict
from .types import ImageEmbedding


def _get_cache_db(model_name: str) -> Path:
    model_name = model_name.replace("/", "_").replace("\\", "_")
    cache_dir = Path.home() / ".cache" / "vism"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model_name}.db"


def _compute_cache_key(path: Path) -> str:
    stat = path.stat()
    key_string = f"{path.absolute()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(key_string.encode()).hexdigest()


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            cache_key TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON embeddings(path)")
    conn.commit()
    return conn


def cache_embeddings(embeddings: List[ImageEmbedding], model_name: str) -> None:
    if not embeddings:
        return

    db_path = _get_cache_db(model_name)

    try:
        conn = _init_db(db_path)
        data = [
            (
                _compute_cache_key(emb.path),
                str(emb.path.absolute()),
                emb.embedding.tobytes(),
            )
            for emb in embeddings
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (cache_key, path, embedding) VALUES (?, ?, ?)",
            data,
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to cache batch of embeddings: {e}")


def load_cached_embeddings(
    paths: List[Path], model_name: str
) -> Dict[Path, ImageEmbedding]:
    db_path = _get_cache_db(model_name)

    path_to_key = {}
    for path in paths:
        path_to_key[_compute_cache_key(path)] = path

    if not path_to_key:
        return {}

    try:
        conn = sqlite3.connect(db_path)
        placeholders = ",".join("?" * len(path_to_key))
        cursor = conn.execute(
            f"SELECT cache_key, embedding FROM embeddings WHERE cache_key IN ({placeholders})",
            tuple(path_to_key.keys()),
        )

        results = {}
        for cache_key, embedding_blob in cursor:
            path = path_to_key[cache_key]
            embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)
            results[path] = ImageEmbedding(path=path, embedding=embedding_array)

        conn.close()
        return results
    except Exception as e:
        print(f"Warning: Failed to load cached embeddings: {e}")
        return {}
