from pathlib import Path
import click
import logging
import sys

MODEL_CHOICES = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
DEFAULT_MODEL = "dinov2_vits14"


def setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging based on verbosity flags"""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )


@click.group(no_args_is_help=True)
@click.option(
    "-v", "--verbose", is_flag=True, default=False, help="Enable debug output"
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress all output except errors",
)
@click.pass_context
def vism(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """vism: Visual Search CLI"""
    ctx.ensure_object(dict)
    setup_logging(verbose, quiet)


@vism.command(no_args_is_help=True)
@click.argument(
    "source_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("query", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-m",
    "--model",
    default=DEFAULT_MODEL,
    type=click.Choice(MODEL_CHOICES),
    show_default=True,
    help="DINOv2 model variant to use for embeddings",
)
@click.option(
    "-k",
    "--limit",
    default=10,
    type=int,
    help="Number of top matches to return",
)
@click.option(
    "-o",
    "--open-with",
    type=str,
    default=None,
    help="Open results with specified application",
)
def search(
    source_dir: Path, query: Path, model: str, limit: int, open_with: str
) -> None:
    """
    Search for images similar to query image in source directory
    """
    from .embeddings import load_model
    from .core import run_search_pipeline

    loaded_model = load_model(model)

    results = run_search_pipeline(
        source_dir=source_dir,
        query=query,
        model=loaded_model,
        model_name=model,
        k=limit,
    )

    if results:
        click.echo("\nTop matches:")
        for result in results:
            click.echo(f"{result.score:.4f} → {result.path}")
        if limit > 0 and results and open_with:
            import subprocess

            subprocess.Popen(
                [open_with] + [str(res.path) for res in results],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    else:
        click.echo("Search failed or returned no results")


def main() -> None:
    vism()


if __name__ == "__main__":
    main()
