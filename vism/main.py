import sys
from pathlib import Path
from tqdm import tqdm

from .images import find_images_recursive, load_image
from .embeddings import load_model, encode_image, encode_images
from .search import build_index, search_items
from .cache import cache_embedding, load_cached_embedding


def main():
    if len(sys.argv) < 2:
        print("Usage: vism <dir> <query_image>")
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    query = Path(sys.argv[2])

    image_paths = list(find_images_recursive(source_dir))
    print(f"Found {len(image_paths)} images")

    model_name = "dinov2_vits14"
    model = load_model(model_name)

    embeddings = []
    batch_size = 64
    cache_hits = 0

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i : i + batch_size]
        cached_items = []
        uncached_paths = []

        for path in batch_paths:
            cached = load_cached_embedding(path, model_name)
            if cached is not None:
                cached_items.append(cached)
            else:
                uncached_paths.append(path)

        embeddings.extend(cached_items)
        cache_hits += len(cached_items)

        if uncached_paths:
            imgs = [load_image(path) for path in uncached_paths]
            batch_embeddings = encode_images(imgs, model)
            embeddings.extend(batch_embeddings)
            for emb in batch_embeddings:
                cache_embedding(emb, model_name)

    print(f"Cache hits: {cache_hits}/{len(image_paths)}")
    index = build_index(embeddings)

    print("Encoding query image...")
    query_img = load_image(query)
    query_embedding = encode_image(query_img, model)

    if query_embedding.embedding is not None:
        results = search_items(index, query_embedding, embeddings, k=10)
        print("\nTop matches:")
        for result in results:
            print(f"{result.score:.4f} → {result.path}")
    else:
        print("Failed to encode query image")


if __name__ == "__main__":
    main()
