import torch
import numpy as np
from typing import List, Tuple, cast
from torchvision import transforms
from .types import ImageData, ImageEmbedding

Model = Tuple[torch.nn.Module, transforms.Compose, str]


def load_model(name: str) -> Model:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINOv2 {name} → {device}")
    model = cast(torch.nn.Module, torch.hub.load("facebookresearch/dinov2", name))
    model.eval()
    model = model.to(device)
    # DINOv2 preprocessing
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
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
