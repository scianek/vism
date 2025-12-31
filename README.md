# vism - Visual Similarity Search CLI

A simple command-line tool for finding visually similar images in a directory using **DINOv2** embeddings and **FAISS** for fast similarity search.

It computes semantic image embeddings with DINOv2, caches them locally, and lets you quickly search for the most similar images to a given query image.

## Features

- Recursive search for images in a directory
- Uses state-of-the-art DINOv2 embeddings for semantic similarity
- Caches embeddings in SQLite to avoid recomputing on unchanged files
- Fast nearest-neighbor search with FAISS

## Installation

This project uses **uv** for dependency and package management.

### Install with uv (recommended)

From the local project directory (where `pyproject.toml` is located):
```bash
uv tool install .
```
This installs the `vism` command globally in an isolated environment.

### Alternative: Using pipx (for users without uv)

`pipx` is a popular tool for installing Python CLI applications in isolated environments (similar to `uv tool install`).
From the local project directory:

1. Build the package wheel:
```bash
python -m build
```

2. Install the generated wheel:
```bash
pipx install dist/vism-*.whl
```

This makes the `vism` command available globally.

### Alternative: Manual installation with pip (not recommended for global use)

If you prefer not to use `uv` or `pipx`:
```bash
git clone https://github.com/scianek/vism.git
cd vism
python -m venv .venv
source .venv/bin/activate
pip install .
```

The `vism` command will be available while the virtual environment is activated. For global access without a tool like `pipx`, you can install into your user site-packages with `pip install --user .`, but this may lead to dependency conflicts.
