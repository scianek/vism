import sys
from pathlib import Path
from tqdm import tqdm

from .images import find_images_recursive, load_image
from .embeddings import load_model, encode_image, encode_images
from .search import build_index, search_items


def main():
    if len(sys.argv) < 2:
        print("Usage: vism <dir> <query_image>")
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    query = Path(sys.argv[2])

    image_paths = find_images_recursive(source_dir)
    model = load_model("dinov2_vits14")
    embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i : i + batch_size]
        imgs = [load_image(path) for path in batch_paths]
        batch_embeddings = encode_images(imgs, model)
        embeddings.extend(batch_embeddings)
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
