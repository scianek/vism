import torch
import numpy as np
from typing import List, Tuple, cast
from torchvision import transforms
import logging
from .types import ImageData, ImageEmbedding

logger = logging.getLogger(__name__)

Model = Tuple[torch.nn.Module, transforms.Compose, str]


def _dinov2_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class _ClipWrapper(torch.nn.Module):
    """Thin wrapper so CLIP encode_image works like a standard forward() call."""

    def __init__(self, clip_model: torch.nn.Module) -> None:
        super().__init__()
        self._clip = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._clip.encode_image(x)


def _parse_clip_name(name: str) -> Tuple[str, str]:
    """'clip_ViT-B-32_openai' → ('ViT-B-32', 'openai')"""
    parts = name.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid CLIP model name '{name}'. Expected clip_<arch>_<pretrained>")
    return parts[1], parts[2]


def load_model(name: str) -> Model:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if name.startswith("clip_"):
        import open_clip

        arch, pretrained = _parse_clip_name(name)
        logger.debug(f"Loading CLIP {arch}/{pretrained} → {device}")
        clip_model, _, preprocess_pil = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained
        )
        clip_model.eval()
        clip_model = clip_model.to(device)
        model: torch.nn.Module = _ClipWrapper(clip_model)
        # open_clip returns a torchvision-compatible transform; wrap it so our
        # pipeline can handle PIL Images the same way as DINOv2.
        preprocess = transforms.Compose([preprocess_pil])
    else:
        logger.debug(f"Loading DINOv2 {name} → {device}")
        model = cast(torch.nn.Module, torch.hub.load("facebookresearch/dinov2", name))
        model.eval()
        model = model.to(device)
        preprocess = _dinov2_preprocess()

    return model, preprocess, device


def _compute_embeddings(
    batch_tensor: torch.Tensor, model_dino: torch.nn.Module
) -> np.ndarray:
    with torch.no_grad():
        embeddings = model_dino(batch_tensor)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().numpy().astype(np.float32)


def encode_image(img: ImageData, model: Model) -> ImageEmbedding:
    model_dino, preprocess, device = model
    processed_image = cast(torch.Tensor, preprocess(img.image)).unsqueeze(0).to(device)
    embedding = _compute_embeddings(processed_image, model_dino)[0]
    return ImageEmbedding(path=img.path, embedding=embedding)


def encode_images(imgs: List[ImageData], model: Model) -> List[ImageEmbedding]:
    model_dino, preprocess, device = model
    processed_images = [cast(torch.Tensor, preprocess(img.image)) for img in imgs]
    batch_tensor = torch.stack(processed_images).to(device)
    embeddings_array = _compute_embeddings(batch_tensor, model_dino)
    return [
        ImageEmbedding(path=img.path, embedding=embedding)
        for img, embedding in zip(imgs, embeddings_array)
    ]
