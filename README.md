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

```bash
vism search [OPTIONS]
```

**Arguments:**

- `source_dir` - Directory to search (recursively scanned for images)
- `query_image` - Path to the query image

**Options:**

- `-k`, `--limit` - Number of top matches to return (default: `10`)

**Example:**

```bash
vism search ~/photos/ ~/query-photo.jpg -k 5
```
