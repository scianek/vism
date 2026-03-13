# vism - Visual Similarity Search CLI

A command-line tool for finding visually similar images using DINOv2 embeddings and FAISS. Point it at a directory and a query image - it encodes everything with DINOv2, caches embeddings locally in SQLite, and returns the closest matches ranked by cosine similarity.

## Features

- Recursive image discovery
- DINOv2 embeddings for semantic visual similarity
- SQLite embedding cache - skips recomputation for unchanged files
- FAISS inner-product index for fast nearest-neighbor search
- GPU support if CUDA is available, otherwise CPU

## Installation

Requires Python 3.10+. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/scianek/vism
```

Or clone and install locally:

```bash
git clone https://github.com/scianek/vism
cd vism
uv sync
```

## Usage
 
### search
 
```bash
vism search <source_dir> <query_image> [OPTIONS]
```
 
**Options:**

- `-m`, `--model` - DINOv2 model variant: `dinov2_vits14` (default), `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`
- `-k`, `--limit` - Number of top matches to return (default: `10`)
- `-o`, `--open-with` - Open results with specified application

**Example:**
 
```bash
vism search ~/photos/ ~/query-photo.jpg -k 5 -o imv -m dinov2_vitl14
```
 
### index

```bash
vism index <source_dir> [-m MODEL]
```

Pre-compute and cache embeddings for all images in a directory without running a search. Useful for indexing a new photo library in the background so subsequent searches are instant.

**Example:**

```bash
vism index ~/photos/
```

### cache
 
Manage the local embeddings cache.
 
#### stats
 
```bash
vism cache stats [PREFIX] [-m MODEL]
```
 
Show cache stats. Without `PREFIX`, prints total entry count per model. With `PREFIX`, shows cached/total coverage per immediate subdirectory - useful for checking whether a specific album has been indexed.
 
#### prune
 
```bash
vism cache prune [PREFIX] [-m MODEL]
```
 
Remove entries for files that no longer exist on disk. Useful after moving or deleting photos.
 
#### clear
 
```bash
vism cache clear [PREFIX] [-m MODEL]
```
 
Delete cache entries. Without arguments clears everything; with `PREFIX` removes only entries under that directory.
 
All cache commands accept an optional `-m`/`--model` flag to target a specific model's cache. Without it, the command operates across all models.
 
## Supported Image Formats
 
`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff` (case-insensitive)
 
## Embedding Cache
 
Embeddings are cached in SQLite at `~/.cache/vism/<model>.db`. Cache keys are derived from the file path, size, and modification time - so the cache is automatically invalidated when a file changes.
