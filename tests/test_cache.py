import numpy as np
import pytest
from pathlib import Path

from vism.cache import (
    _compute_cache_key,
    _decode_path,
    cache_embeddings,
    load_cached_embeddings,
    clear_cache,
    prune_cache,
    stats_cache_global,
    stats_cache_prefix,
)
from vism.types import ImageEmbedding


def make_embedding(path: Path, values: list[float]) -> ImageEmbedding:
    return ImageEmbedding(path=path, embedding=np.array(values, dtype=np.float32))


@pytest.fixture
def tmp_cache(tmp_path: Path, monkeypatch):
    """Redirect cache directory to a temp path"""
    cache_dir = tmp_path / ".cache" / "vism"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return cache_dir


# --- _compute_cache_key ---


def test_cache_key_same_file(tmp_path: Path):
    f = tmp_path / "img.jpg"
    f.write_bytes(b"data")
    assert _compute_cache_key(f) == _compute_cache_key(f)


def test_cache_key_changes_on_content(tmp_path: Path):
    f = tmp_path / "img.jpg"
    f.write_bytes(b"data")
    key1 = _compute_cache_key(f)
    f.write_bytes(b"different data")
    key2 = _compute_cache_key(f)
    assert key1 != key2


def test_cache_key_different_files(tmp_path: Path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    assert _compute_cache_key(a) != _compute_cache_key(b)


# --- _decode_path ---


def test_decode_path_roundtrip(tmp_path: Path):
    path = tmp_path / "some" / "image.jpg"
    encoded = str(path.absolute()).encode(errors="surrogateescape")
    assert _decode_path(encoded) == path


def test_decode_path_with_spaces(tmp_path: Path):
    path = tmp_path / "my photos" / "image file.jpg"
    encoded = str(path.absolute()).encode(errors="surrogateescape")
    assert _decode_path(encoded) == path


# --- cache_embeddings + load_cached_embeddings ---


def test_cache_and_load_roundtrip(
    tmp_path: Path,
):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"x")
    emb = make_embedding(img, [0.1, 0.2, 0.3])

    cache_embeddings([emb], "test_model")
    result = load_cached_embeddings([img], "test_model")

    assert img in result
    np.testing.assert_array_almost_equal(result[img].embedding, emb.embedding)


def test_load_returns_empty_for_unknown_file(
    tmp_path: Path,
):
    img = tmp_path / "ghost.jpg"
    img.write_bytes(b"x")
    result = load_cached_embeddings([img], "test_model")
    assert result == {}


def test_cache_multiple_embeddings(
    tmp_path: Path,
):
    paths = []
    embs = []
    for i in range(5):
        p = tmp_path / f"img{i}.jpg"
        p.write_bytes(f"data{i}".encode())
        paths.append(p)
        embs.append(make_embedding(p, [float(i), float(i + 1)]))

    cache_embeddings(embs, "test_model")
    result = load_cached_embeddings(paths, "test_model")

    assert len(result) == 5
    for p, e in zip(paths, embs):
        np.testing.assert_array_almost_equal(result[p].embedding, e.embedding)


def test_cache_overwrites_existing(
    tmp_path: Path,
):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"x")

    emb1 = make_embedding(img, [1.0, 0.0])
    emb2 = make_embedding(img, [0.0, 1.0])

    cache_embeddings([emb1], "test_model")
    cache_embeddings([emb2], "test_model")

    result = load_cached_embeddings([img], "test_model")
    np.testing.assert_array_almost_equal(result[img].embedding, emb2.embedding)


# --- clear_cache ---


def test_clear_all(
    tmp_path: Path,
):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"x")
    cache_embeddings([make_embedding(img, [1.0])], "test_model")

    deleted = clear_cache()
    assert deleted == 1

    result = load_cached_embeddings([img], "test_model")
    assert result == {}


def test_clear_specific_model(
    tmp_path: Path,
):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"x")
    cache_embeddings([make_embedding(img, [1.0])], "model_a")
    cache_embeddings([make_embedding(img, [2.0])], "model_b")

    deleted = clear_cache(model_name="model_a")
    assert deleted == 1

    assert load_cached_embeddings([img], "model_a") == {}
    assert img in load_cached_embeddings([img], "model_b")


def test_clear_with_prefix(
    tmp_path: Path,
):
    keep = tmp_path / "keep" / "img.jpg"
    remove = tmp_path / "remove" / "img.jpg"
    keep.parent.mkdir()
    remove.parent.mkdir()
    keep.write_bytes(b"k")
    remove.write_bytes(b"r")

    cache_embeddings(
        [make_embedding(keep, [1.0]), make_embedding(remove, [2.0])], "test_model"
    )

    deleted = clear_cache(prefix=tmp_path / "remove")
    assert deleted == 1

    assert keep in load_cached_embeddings([keep], "test_model")
    assert load_cached_embeddings([remove], "test_model") == {}


# --- prune_cache ---


def test_prune_removes_missing_files(
    tmp_path: Path,
):
    existing = tmp_path / "exists.jpg"
    missing = tmp_path / "gone.jpg"
    existing.write_bytes(b"x")
    missing.write_bytes(b"x")

    cache_embeddings(
        [make_embedding(existing, [1.0]), make_embedding(missing, [2.0])],
        "test_model",
    )

    missing.unlink()
    pruned = prune_cache()

    assert pruned == 1
    assert existing in load_cached_embeddings([existing], "test_model")


def test_prune_keeps_existing_files(
    tmp_path: Path,
):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"x")
    cache_embeddings([make_embedding(img, [1.0])], "test_model")

    pruned = prune_cache()
    assert pruned == 0
    assert img in load_cached_embeddings([img], "test_model")


def test_prune_with_prefix(
    tmp_path: Path,
):
    kept_dir = tmp_path / "kept"
    pruned_dir = tmp_path / "pruned"
    kept_dir.mkdir()
    pruned_dir.mkdir()

    kept = kept_dir / "img.jpg"
    gone = pruned_dir / "img.jpg"
    kept.write_bytes(b"k")
    gone.write_bytes(b"g")

    cache_embeddings(
        [make_embedding(kept, [1.0]), make_embedding(gone, [2.0])],
        "test_model",
    )

    gone.unlink()
    # prune scoped to kept_dir - gone is outside prefix, should be untouched
    pruned = prune_cache(prefix=kept_dir)
    assert pruned == 0


# --- stats_cache_global ---


def test_stats_global_empty():
    assert stats_cache_global() == {}


def test_stats_global_counts(
    tmp_path: Path,
):
    for i in range(3):
        p = tmp_path / f"img{i}.jpg"
        p.write_bytes(f"d{i}".encode())
        cache_embeddings([make_embedding(p, [float(i)])], "test_model")

    result = stats_cache_global()
    assert result["test_model"] == 3


def test_stats_global_per_model(
    tmp_path: Path,
):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"x")
    cache_embeddings([make_embedding(img, [1.0])], "model_a")
    cache_embeddings([make_embedding(img, [2.0])], "model_b")

    result = stats_cache_global()
    assert result["model_a"] == 1
    assert result["model_b"] == 1


# --- stats_cache_prefix ---


def test_stats_prefix_cached_and_total(
    tmp_path: Path,
):
    album = tmp_path / "album"
    sub_a = album / "sub_a"
    sub_b = album / "sub_b"
    sub_a.mkdir(parents=True)
    sub_b.mkdir(parents=True)

    # 2 images in sub_a, 1 in sub_b
    imgs_a = []
    for i in range(2):
        p = sub_a / f"img{i}.jpg"
        p.write_bytes(f"d{i}".encode())
        imgs_a.append(p)

    img_b = sub_b / "img.jpg"
    img_b.write_bytes(b"d")

    # only cache sub_a
    cache_embeddings(
        [make_embedding(p, [float(i)]) for i, p in enumerate(imgs_a)], "test_model"
    )

    result = stats_cache_prefix(album, model_name="test_model")
    counts = result["test_model"]

    assert counts[str(sub_a)] == (2, 2)
    assert counts[str(sub_b)] == (0, 1)


def test_stats_prefix_excludes_outside_paths(
    tmp_path: Path,
):
    inside = tmp_path / "inside" / "img.jpg"
    outside = tmp_path / "outside" / "img.jpg"
    inside.parent.mkdir()
    outside.parent.mkdir()
    inside.write_bytes(b"i")
    outside.write_bytes(b"o")

    cache_embeddings(
        [make_embedding(inside, [1.0]), make_embedding(outside, [2.0])],
        "test_model",
    )

    result = stats_cache_prefix(tmp_path / "inside", model_name="test_model")
    counts = result["test_model"]

    # outside should not appear
    assert all("outside" not in k for k in counts)
