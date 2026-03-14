import sqlite3
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging
from .types import ImageEmbedding

logger = logging.getLogger(__name__)


def _get_cache_db(model_name: str) -> Path:
    model_name = model_name.replace("/", "_").replace("\\", "_")
    cache_dir = Path.home() / ".cache" / "vism"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model_name}.db"


def _get_all_cache_dbs() -> Dict[str, Path]:
    """Return all existing model cache dbs as {model_name: path}"""
    cache_dir = Path.home() / ".cache" / "vism"
    if not cache_dir.exists():
        return {}
    return {p.stem: p for p in cache_dir.glob("*.db")}


def _compute_cache_key(path: Path) -> str:
    stat = path.stat()
    key_string = f"{path.absolute()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(key_string.encode(errors="surrogateescape")).hexdigest()


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            cache_key TEXT PRIMARY KEY,
            path BLOB NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON embeddings(path)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS failed (
            cache_key TEXT PRIMARY KEY,
            path BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def _iter_dbs(model_name: Optional[str]) -> Dict[str, Path]:
    """Return {model_name: db_path} for the given model or all models"""
    if model_name is not None:
        db_path = _get_cache_db(model_name)
        return {model_name: db_path} if db_path.exists() else {}
    return _get_all_cache_dbs()


def _decode_path(path_blob: bytes) -> Path:
    return Path(path_blob.decode(errors="surrogateescape"))


def cache_embeddings(embeddings: List[ImageEmbedding], model_name: str) -> None:
    if not embeddings:
        return

    db_path = _get_cache_db(model_name)

    valid_data_rows = []

    for emb in embeddings:
        try:
            cache_key = _compute_cache_key(emb.path)

            path_bytes = str(emb.path.absolute()).encode(errors="surrogateescape")

            embedding_blob = emb.embedding.tobytes()

            valid_data_rows.append((cache_key, path_bytes, embedding_blob))

        except Exception as e:
            logger.warning(
                f"Skipping cache for file '{emb.path.name}' due to preparation error: {e}"
            )
            continue

    if not valid_data_rows:
        return

    try:
        conn = _init_db(db_path)
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (cache_key, path, embedding) VALUES (?, ?, ?)",
            valid_data_rows,
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to execute cache transaction for batch: {e}")


def load_cached_embeddings(
    paths: List[Path], model_name: str
) -> Dict[Path, ImageEmbedding]:
    db_path = _get_cache_db(model_name)

    path_to_key = {}
    for path in paths:
        try:
            path_to_key[_compute_cache_key(path)] = path
        except Exception as e:
            logger.warning(
                f"Skipping file '{path.name}' during cache lookup due to error: {e}"
            )
            continue

    if not path_to_key:
        return {}

    try:
        conn = sqlite3.connect(db_path)
        results = {}
        keys = list(path_to_key.keys())
        for batch_start in range(0, len(keys), 999):
            batch = keys[batch_start : batch_start + 999]
            placeholders = ",".join("?" * len(batch))
            cursor = conn.execute(
                f"SELECT cache_key, embedding FROM embeddings WHERE cache_key IN ({placeholders})",
                batch,
            )
            for cache_key, embedding_blob in cursor:
                path = path_to_key[cache_key]
                embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)
                results[path] = ImageEmbedding(path=path, embedding=embedding_array)
        conn.close()
        return results
    except Exception as e:
        logger.warning(f"Failed to load cached embeddings: {e}")
        return {}


def mark_failed(path: Path, model_name: str) -> None:
    """Record a path as permanently failed so it is skipped on future runs"""
    db_path = _get_cache_db(model_name)
    try:
        cache_key = _compute_cache_key(path)
        path_bytes = str(path.absolute()).encode(errors="surrogateescape")
        conn = _init_db(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO failed (cache_key, path) VALUES (?, ?)",
            (cache_key, path_bytes),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to record failed path '{path.name}': {e}")


def load_failed_paths(paths: List[Path], model_name: str) -> set[Path]:
    """Return the subset of paths that are recorded as failed"""
    db_path = _get_cache_db(model_name)
    if not db_path.exists():
        return set()

    path_to_key: dict[str, Path] = {}
    for path in paths:
        try:
            path_to_key[_compute_cache_key(path)] = path
        except Exception:
            continue

    if not path_to_key:
        return set()

    try:
        conn = sqlite3.connect(db_path)
        keys = list(path_to_key.keys())
        result = set()
        for batch_start in range(0, len(keys), 999):
            batch = keys[batch_start : batch_start + 999]
            placeholders = ",".join("?" * len(batch))
            cursor = conn.execute(
                f"SELECT cache_key FROM failed WHERE cache_key IN ({placeholders})",
                batch,
            )
            result.update(path_to_key[row[0]] for row in cursor)
        conn.close()
        return result
    except Exception as e:
        logger.warning(f"Failed to load failed paths: {e}")
        return set()


def clear_cache(model_name: Optional[str] = None, prefix: Optional[Path] = None) -> int:
    dbs = _iter_dbs(model_name)
    total_deleted = 0

    for name, db_path in dbs.items():
        if not db_path.exists():
            continue
        try:
            conn = _init_db(db_path)
            if prefix is None:
                cursor = conn.execute("DELETE FROM embeddings")
                total_deleted += cursor.rowcount
            else:
                prefix_str = str(prefix.absolute())
                cursor = conn.execute("SELECT cache_key, path FROM embeddings")
                keys_to_delete = [
                    row[0]
                    for row in cursor
                    if str(_decode_path(row[1])).startswith(prefix_str)
                ]
                if keys_to_delete:
                    conn.execute(
                        f"DELETE FROM embeddings WHERE cache_key IN ({','.join('?' * len(keys_to_delete))})",
                        keys_to_delete,
                    )
                    total_deleted += len(keys_to_delete)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to clear cache for model '{name}': {e}")

    return total_deleted


def prune_cache(model_name: Optional[str] = None, prefix: Optional[Path] = None) -> int:
    dbs = _iter_dbs(model_name)
    total_pruned = 0

    for name, db_path in dbs.items():
        if not db_path.exists():
            continue
        try:
            conn = _init_db(db_path)
            cursor = conn.execute("SELECT cache_key, path FROM embeddings")
            rows = cursor.fetchall()

            keys_to_delete = []
            prefix_str = str(prefix.absolute()) if prefix is not None else None

            for cache_key, path_blob in rows:
                path = _decode_path(path_blob)
                if prefix_str is not None and not str(path).startswith(prefix_str):
                    continue
                if not path.exists():
                    keys_to_delete.append(cache_key)

            if keys_to_delete:
                conn.execute(
                    f"DELETE FROM embeddings WHERE cache_key IN ({','.join('?' * len(keys_to_delete))})",
                    keys_to_delete,
                )
                total_pruned += len(keys_to_delete)

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to prune cache for model '{name}': {e}")

    return total_pruned


def stats_cache_global(model_name: Optional[str] = None) -> Dict[str, int]:
    dbs = _iter_dbs(model_name)
    results: Dict[str, int] = {}

    for name, db_path in dbs.items():
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(db_path)
            (count,) = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            conn.close()
            results[name] = count
        except Exception as e:
            logger.warning(f"Failed to read stats for model '{name}': {e}")

    return results


def stats_cache_prefix(
    prefix: Path, model_name: Optional[str] = None
) -> Dict[str, Dict[str, tuple[int, int]]]:
    from .images import find_images_recursive

    prefix_abs = prefix.absolute()

    total_counts: Dict[str, int] = {}
    try:
        all_images = find_images_recursive(prefix_abs)
        for img_path in all_images:
            try:
                rel = img_path.relative_to(prefix_abs)
                group = str(prefix_abs / rel.parts[0])
                total_counts[group] = total_counts.get(group, 0) + 1
            except ValueError:
                continue
    except Exception as e:
        logger.warning(f"Failed to scan directory '{prefix}': {e}")

    dbs = _iter_dbs(model_name)
    results: Dict[str, Dict[str, tuple[int, int]]] = {}

    for name, db_path in dbs.items():
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT path FROM embeddings")
            rows = cursor.fetchall()
            conn.close()

            cached_counts: Dict[str, int] = {}
            for (path_blob,) in rows:
                path = _decode_path(path_blob)
                try:
                    rel = path.relative_to(prefix_abs)
                    group = str(prefix_abs / rel.parts[0])
                    cached_counts[group] = cached_counts.get(group, 0) + 1
                except ValueError:
                    continue

            all_groups = set(cached_counts) | set(total_counts)
            if all_groups:
                results[name] = {
                    group: (cached_counts.get(group, 0), total_counts.get(group, 0))
                    for group in all_groups
                }

        except Exception as e:
            logger.warning(f"Failed to read stats for model '{name}': {e}")

    return results
