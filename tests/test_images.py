import pytest
from pathlib import Path

from vism.images import find_images_recursive


SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    return tmp_path


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image data")
    return path


def test_finds_all_supported_extensions(image_dir: Path):
    for ext in SUPPORTED_EXTENSIONS:
        touch(image_dir / f"img.{ext}")

    found = find_images_recursive(image_dir)
    found_exts = {p.suffix.lower().lstrip(".") for p in found}
    assert found_exts == set(SUPPORTED_EXTENSIONS)


def test_case_insensitive(image_dir: Path):
    touch(image_dir / "img.JPG")
    touch(image_dir / "img.PNG")
    found = find_images_recursive(image_dir)
    assert len(found) == 2


def test_ignores_non_images(image_dir: Path):
    touch(image_dir / "doc.txt")
    touch(image_dir / "data.json")
    touch(image_dir / "img.jpg")
    found = find_images_recursive(image_dir)
    assert len(found) == 1
    assert found[0].name == "img.jpg"


def test_recursive_discovery(image_dir: Path):
    touch(image_dir / "a" / "img1.jpg")
    touch(image_dir / "a" / "b" / "img2.png")
    touch(image_dir / "c" / "img3.webp")
    found = find_images_recursive(image_dir)
    assert len(found) == 3


def test_returns_sorted_paths(image_dir: Path):
    touch(image_dir / "c.jpg")
    touch(image_dir / "a.jpg")
    touch(image_dir / "b.jpg")
    found = find_images_recursive(image_dir)
    assert found == sorted(found)


def test_deduplicates_results(image_dir: Path):
    touch(image_dir / "img.jpg")
    found = find_images_recursive(image_dir)
    assert len(found) == len(set(found))


def test_empty_directory(image_dir: Path):
    found = find_images_recursive(image_dir)
    assert found == []


def test_raises_on_non_directory(tmp_path: Path):
    f = tmp_path / "file.txt"
    f.write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        find_images_recursive(f)
