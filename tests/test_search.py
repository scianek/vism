import numpy as np
import pytest
from pathlib import Path

from vism.search import build_index, search_items
from vism.types import ImageEmbedding


def normalized(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def make_embedding(path: Path, vec: list[float]) -> ImageEmbedding:
    return ImageEmbedding(path=path, embedding=normalized(vec))


def test_top_result_is_identical_vector(tmp_path: Path):
    embs = [
        make_embedding(tmp_path / "a.jpg", [1.0, 0.0, 0.0]),
        make_embedding(tmp_path / "b.jpg", [0.0, 1.0, 0.0]),
        make_embedding(tmp_path / "c.jpg", [0.0, 0.0, 1.0]),
    ]
    index = build_index(embs)
    query = make_embedding(tmp_path / "q.jpg", [1.0, 0.0, 0.0])
    results = search_items(index, query, embs, k=1)

    assert len(results) == 1
    assert results[0].path == tmp_path / "a.jpg"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_results_sorted_by_score_desc(tmp_path: Path):
    embs = [
        make_embedding(tmp_path / "a.jpg", [1.0, 0.0]),
        make_embedding(tmp_path / "b.jpg", [1.0, 0.5]),
        make_embedding(tmp_path / "c.jpg", [0.0, 1.0]),
    ]
    index = build_index(embs)
    query = make_embedding(tmp_path / "q.jpg", [1.0, 0.0])
    results = search_items(index, query, embs, k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_k_limits_results(tmp_path: Path):
    embs = [make_embedding(tmp_path / f"img{i}.jpg", [float(i), 1.0]) for i in range(10)]
    index = build_index(embs)
    query = make_embedding(tmp_path / "q.jpg", [1.0, 1.0])
    results = search_items(index, query, embs, k=3)
    assert len(results) == 3


def test_index_total_matches_embeddings(tmp_path: Path):
    embs = [make_embedding(tmp_path / f"img{i}.jpg", [float(i), 1.0]) for i in range(5)]
    index = build_index(embs)
    assert index.ntotal == 5
