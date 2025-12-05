import sys
from pathlib import Path

from .embeddings import load_model
from .core import run_search_pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: vism <dir> <query_image>")
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    query = Path(sys.argv[2])

    model_name = "dinov2_vits14"
    model = load_model(model_name)

    results = run_search_pipeline(
        source_dir=source_dir,
        query=query,
        model=model,
        model_name=model_name,
        k=10,
    )

    if results:
        print("\nTop matches:")
        for result in results:
            print(f"{result.score:.4f} → {result.path}")
    else:
        print("Search failed or returned no results")


if __name__ == "__main__":
    main()
