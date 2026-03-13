from pathlib import Path
import click
import logging
import sys
from typing import Optional

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


@vism.group(no_args_is_help=True)
def cache() -> None:
    """Manage the embeddings cache"""
    pass


@cache.command()
@click.argument("prefix", required=False, default=None, type=click.Path(path_type=Path))
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Choice(MODEL_CHOICES),
    help="Target a specific model's cache (default: all models)",
)
def clear(prefix: Optional[Path], model: Optional[str]) -> None:
    """Remove cache entries. Optionally filter by path prefix and/or model"""
    from .cache import clear_cache

    deleted = clear_cache(model_name=model, prefix=prefix)
    click.echo(f"Cleared {deleted} cache entr{'y' if deleted == 1 else 'ies'}")


@cache.command()
@click.argument("prefix", required=False, default=None, type=click.Path(path_type=Path))
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Choice(MODEL_CHOICES),
    help="Target a specific model's cache (default: all models)",
)
def prune(prefix: Optional[Path], model: Optional[str]) -> None:
    """Remove entries for files that no longer exist on disk"""
    from .cache import prune_cache

    pruned = prune_cache(model_name=model, prefix=prefix)
    click.echo(f"Pruned {pruned} dangling cache entr{'y' if pruned == 1 else 'ies'}")


@cache.command()
@click.argument("prefix", required=False, default=None, type=click.Path(path_type=Path))
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Choice(MODEL_CHOICES),
    help="Target a specific model's cache (default: all models)",
)
def stats(prefix: Optional[Path], model: Optional[str]) -> None:
    """Show cache stats. Without PREFIX shows totals per model; with PREFIX shows coverage per subdirectory"""
    from .cache import stats_cache_global, stats_cache_prefix

    if prefix is None:
        data = stats_cache_global(model_name=model)
        if not data:
            click.echo("No cache entries found")
            return
        for model_name, count in sorted(data.items()):
            click.echo(f"{model_name}: {count} entries")
    else:
        data = stats_cache_prefix(prefix=prefix, model_name=model)
        if not data:
            click.echo("No cache entries found")
            return
        for model_name, counts in sorted(data.items()):
            total_cached = sum(c for c, _ in counts.values())
            total_images = sum(t for _, t in counts.values())
            click.echo(f"\n{model_name} ({total_cached}/{total_images} total):")
            for directory, (cached, total) in sorted(
                counts.items(), key=lambda x: -x[1][0]
            ):
                click.echo(f"  {cached:>6}/{total:<6}  {directory}")


def main() -> None:
    vism()


if __name__ == "__main__":
    main()
