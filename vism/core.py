from pathlib import Path
from typing import List
from .types import ImageEmbedding, SearchResult
from .cache import load_cached_embeddings, cache_embeddings
from .images import load_image, find_images_recursive
from .embeddings import Model, encode_image, encode_images
from .search import build_index, search_items
from tqdm import tqdm


def run_search_pipeline(
    source_dir: Path, query: Path, model: Model, model_name: str, k: int = 10
) -> List[SearchResult]:
    image_paths = list(find_images_recursive(source_dir))
    print(f"Found {len(image_paths)} images")

    embeddings = get_or_compute_embeddings(
        image_paths,
        model,
        model_name,
    )

    print("Building search index...")
    index = build_index(embeddings)

    print("Encoding query image...")
    query_img = load_image(query)
    query_embedding = encode_image(query_img, model)
    return search_items(index, query_embedding, embeddings, k=k)


def get_or_compute_embeddings(
    image_paths: List[Path],
    model: Model,
    model_name: str,
) -> List[ImageEmbedding]:
    embeddings = []
    batch_size = 64

    print("Loading cached embeddings...")
    cached = load_cached_embeddings(
        image_paths,
        model_name,
    )
    embeddings = [cached.get(p) for p in image_paths]
    print(f"Cache hits: {len(cached)}/{len(image_paths)}")

    uncached_indices = [i for i, emb in enumerate(embeddings) if emb is None]

    if uncached_indices:
        print(f"Processing {len(uncached_indices)} uncached images...")
        for batch_start in tqdm(
            range(0, len(uncached_indices), batch_size), desc="Encoding"
        ):
            batch_indices = uncached_indices[batch_start : batch_start + batch_size]
            batch_paths = [image_paths[i] for i in batch_indices]
            imgs = [load_image(path) for path in batch_paths]
            batch_embeddings = encode_images(imgs, model)
            for idx, emb in zip(batch_indices, batch_embeddings):
                embeddings[idx] = emb
            cache_embeddings(batch_embeddings, model_name)

    return [emb for emb in embeddings if emb is not None]
