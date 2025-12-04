import sys
from pathlib import Path
from tqdm import tqdm

from .images import find_images_recursive, load_image
from .embeddings import load_model, encode_image, encode_images
from .search import build_index, search_items
from .cache import cache_embeddings, load_cached_embeddings


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

    print("Loading cached embeddings...")
    cached_embeddings = load_cached_embeddings(
        image_paths,
        model_name,
    )
    cache_hits = len(cached_embeddings)
    print(f"Cache hits: {cache_hits}/{len(image_paths)}")

    uncached_paths = [p for p in image_paths if p not in cached_embeddings]
    embeddings.extend(
        cached_embeddings[p] for p in image_paths if p in cached_embeddings
    )

    if uncached_paths:
        print(f"Processing {len(uncached_paths)} uncached images...")
        for i in tqdm(range(0, len(uncached_paths), batch_size), desc="Encoding"):
            batch_paths = uncached_paths[i : i + batch_size]
            imgs = [load_image(path) for path in batch_paths]
            batch_embeddings = encode_images(imgs, model)
            embeddings.extend(batch_embeddings)
            cache_embeddings(batch_embeddings, model_name)

    path_to_embedding = {emb.path: emb for emb in embeddings}
    embeddings = [path_to_embedding[p] for p in image_paths]

    print("Building search index...")
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
