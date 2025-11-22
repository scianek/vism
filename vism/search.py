import faiss
import numpy as np
from typing import List
from .types import ImageEmbedding, SearchResult


def build_index(items: List[ImageEmbedding]) -> faiss.IndexFlatIP:
    vectors = np.stack([it.embedding for it in items])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))  # type: ignore
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def search_items(
    index: faiss.IndexFlatIP,
    query_embedding: ImageEmbedding,
    embeddings: List[ImageEmbedding],
    k: int = 10,
) -> List[SearchResult]:
    scores, indices = index.search(query_embedding.embedding.reshape(1, -1).astype("float32"), k)  # type: ignore
    return [
        SearchResult(path=embeddings[i].path, score=float(scores[0][j]))
        for j, i in enumerate(indices[0])
        if i != -1
    ]
