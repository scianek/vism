from pathlib import Path
import click


@click.group(no_args_is_help=True)
def vism():
    """vism: Visual Search CLI"""
    pass


@vism.command(no_args_is_help=True)
@click.argument(
    "source_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("query", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-k",
    "--limit",
    default=10,
    type=int,
    help="Number of top matches to return",
)
def search(source_dir: Path, query: Path, limit: int) -> None:
    """
    Search for images similar to query image in source directory
    """
    from .embeddings import load_model
    from .core import run_search_pipeline

    model_name = "dinov2_vits14"
    model = load_model(model_name)

    results = run_search_pipeline(
        source_dir=source_dir,
        query=query,
        model=model,
        model_name=model_name,
        k=limit,
    )

    if results:
        click.echo("\nTop matches:")
        for result in results:
            click.echo(f"{result.score:.4f} → {result.path}")
    else:
        click.echo("Search failed or returned no results")


def main() -> None:
    vism()


if __name__ == "__main__":
    main()
