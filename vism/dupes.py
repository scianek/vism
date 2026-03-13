import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple
import logging
from .types import ImageEmbedding

logger = logging.getLogger(__name__)

# (image_index, max_similarity_to_cluster)
ClusterMember = Tuple[int, float]
Cluster = List[ClusterMember]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        self.parent[self.find(x)] = self.find(y)

    def groups(self) -> List[List[int]]:
        from collections import defaultdict

        buckets: dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            buckets[self.find(i)].append(i)
        return [g for g in buckets.values() if len(g) > 1]


def find_duplicates(
    embeddings: List[ImageEmbedding],
    threshold: float = 0.95,
    k: int = 64,
) -> List[List[Tuple[Path, float]]]:
    """
    Find clusters of near-duplicate images using FAISS + union-find

    Returns a list of clusters, each cluster being a list of
    (path, max_similarity_to_any_other_cluster_member) sorted by score desc
    Clusters are sorted by size desc
    """
    if len(embeddings) < 2:
        return []

    vectors = np.stack([e.embedding for e in embeddings]).astype("float32")
    n = len(vectors)
    k = min(k, n)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)  # type: ignore

    scores, indices = index.search(vectors, k)  # type: ignore

    uf = _UnionFind(n)
    # max similarity each node has to any neighbor above threshold
    max_sim: List[float] = [0.0] * n

    for i in range(n):
        for j_pos in range(1, k):  # skip self at position 0
            j = int(indices[i, j_pos])
            if j == -1:
                break
            sim = float(scores[i, j_pos])
            if sim < threshold:
                break  # scores are descending, no point continuing
            uf.union(i, j)
            if sim > max_sim[i]:
                max_sim[i] = sim
            if sim > max_sim[j]:
                max_sim[j] = sim

    clusters: List[List[Tuple[Path, float]]] = []
    for group in uf.groups():
        members = sorted(
            [(embeddings[i].path, max_sim[i]) for i in group],
            key=lambda x: -x[1],
        )
        clusters.append(members)

    return sorted(clusters, key=lambda c: -len(c))
