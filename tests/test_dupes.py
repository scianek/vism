import numpy as np
import pytest
from pathlib import Path

from vism.dupes import _UnionFind, find_duplicates
from vism.types import ImageEmbedding


def normalized(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def make_embedding(path: Path, vec: list[float]) -> ImageEmbedding:
    return ImageEmbedding(path=path, embedding=normalized(vec))


# --- _UnionFind ---


def test_union_find_initially_separate():
    uf = _UnionFind(3)
    assert uf.find(0) != uf.find(1)
    assert uf.find(1) != uf.find(2)


def test_union_find_merges():
    uf = _UnionFind(3)
    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)
    assert uf.find(0) != uf.find(2)


def test_union_find_transitivity():
    uf = _UnionFind(3)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)


def test_union_find_groups_filters_singletons():
    uf = _UnionFind(4)
    uf.union(0, 1)
    # 2 and 3 are singletons
    groups = uf.groups()
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1]


def test_union_find_multiple_groups():
    uf = _UnionFind(6)
    uf.union(0, 1)
    uf.union(2, 3)
    uf.union(3, 4)
    groups = sorted([sorted(g) for g in uf.groups()])
    assert groups == [[0, 1], [2, 3, 4]]


# --- find_duplicates ---


def test_identical_vectors_cluster(tmp_path: Path):
    vec = [1.0, 0.0, 0.0]
    embs = [make_embedding(tmp_path / f"img{i}.jpg", vec) for i in range(3)]
    clusters = find_duplicates(embs, threshold=0.99)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_dissimilar_vectors_no_clusters(tmp_path: Path):
    embs = [
        make_embedding(tmp_path / "a.jpg", [1.0, 0.0, 0.0]),
        make_embedding(tmp_path / "b.jpg", [0.0, 1.0, 0.0]),
        make_embedding(tmp_path / "c.jpg", [0.0, 0.0, 1.0]),
    ]
    clusters = find_duplicates(embs, threshold=0.99)
    assert clusters == []


def test_two_separate_clusters(tmp_path: Path):
    embs = [
        make_embedding(tmp_path / "a1.jpg", [1.0, 0.0, 0.0]),
        make_embedding(tmp_path / "a2.jpg", [1.0, 0.01, 0.0]),
        make_embedding(tmp_path / "b1.jpg", [0.0, 0.0, 1.0]),
        make_embedding(tmp_path / "b2.jpg", [0.0, 0.01, 1.0]),
    ]
    clusters = find_duplicates(embs, threshold=0.99)
    assert len(clusters) == 2
    paths_per_cluster = [set(p for p, _ in c) for c in clusters]
    assert {tmp_path / "a1.jpg", tmp_path / "a2.jpg"} in paths_per_cluster
    assert {tmp_path / "b1.jpg", tmp_path / "b2.jpg"} in paths_per_cluster


def test_transitivity_clusters_indirectly_similar(tmp_path: Path):
    # a is similar to b, b is similar to c, a and c may not be directly compared
    # but all three should end up in one cluster
    a = normalized([1.0, 0.0, 0.0])
    b = normalized([1.0, 0.05, 0.0])
    c = normalized([1.0, 0.1, 0.0])

    embs = [
        ImageEmbedding(path=tmp_path / "a.jpg", embedding=a),
        ImageEmbedding(path=tmp_path / "b.jpg", embedding=b),
        ImageEmbedding(path=tmp_path / "c.jpg", embedding=c),
    ]
    clusters = find_duplicates(embs, threshold=0.99)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_scores_are_max_similarity(tmp_path: Path):
    embs = [
        make_embedding(tmp_path / "a.jpg", [1.0, 0.0]),
        make_embedding(tmp_path / "b.jpg", [1.0, 0.0]),
    ]
    clusters = find_duplicates(embs, threshold=0.99)
    assert len(clusters) == 1
    for _, score in clusters[0]:
        assert score > 0.99


def test_clusters_sorted_by_size_desc(tmp_path: Path):
    # cluster of 3 and cluster of 2
    embs = [
        make_embedding(tmp_path / "a1.jpg", [1.0, 0.0, 0.0]),
        make_embedding(tmp_path / "a2.jpg", [1.0, 0.01, 0.0]),
        make_embedding(tmp_path / "a3.jpg", [1.0, 0.02, 0.0]),
        make_embedding(tmp_path / "b1.jpg", [0.0, 0.0, 1.0]),
        make_embedding(tmp_path / "b2.jpg", [0.0, 0.01, 1.0]),
    ]
    clusters = find_duplicates(embs, threshold=0.99)
    assert len(clusters) == 2
    assert len(clusters[0]) >= len(clusters[1])


def test_fewer_than_two_images_returns_empty(tmp_path: Path):
    embs = [make_embedding(tmp_path / "a.jpg", [1.0, 0.0])]
    clusters = find_duplicates(embs, threshold=0.95)
    assert clusters == []


def test_threshold_controls_grouping(tmp_path: Path):
    # these vectors are similar but not identical
    embs = [
        make_embedding(tmp_path / "a.jpg", [1.0, 0.0]),
        make_embedding(tmp_path / "b.jpg", [1.0, 0.1]),
    ]
    # strict threshold - no cluster
    assert find_duplicates(embs, threshold=0.9999) == []
    # loose threshold - one cluster
    assert len(find_duplicates(embs, threshold=0.5)) == 1
