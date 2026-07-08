"""Image loading & preprocessing without torchvision / cv2 (PIL + torch only)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

# DINOv3 / DINOv2 use standard ImageNet normalization statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PATCH_SIZE = 16  # dinov3 vit*/16

# 90도 배수 회전(무손실: 보간/빈모서리 없음) — memory bank 증강용.
_ROT90 = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def rotate90(img: Image.Image, deg: int) -> Image.Image:
    """PIL 이미지를 90도 배수로 정확히 회전 (deg in {0,90,180,270})."""
    deg = int(deg) % 360
    if deg == 0:
        return img
    if deg not in _ROT90:
        raise ValueError(f"rotate90 은 90도 배수만 지원 (입력 {deg}).")
    return img.transpose(_ROT90[deg])


def rotate_image(img: Image.Image, deg: float) -> Image.Image:
    """임의 각도 회전 (memory bank 증강용).

      - 90도 배수: transpose 로 **무손실**(보간/빈모서리 없음).
      - 그 외(예: 45도): bilinear 보간 + **reflect 패딩**(가장자리 반사)으로 빈 모서리
        채움 → 검은 잔상 없이 텍스처가 자연스럽게 이어짐 (AnomalyDINO 의 BORDER_REFLECT_101).
    """
    d = float(deg) % 360.0
    if d == 0.0:
        return img
    if d % 90.0 == 0.0:
        return rotate90(img, int(d))
    from skimage.transform import rotate as sk_rotate   # 비직각 회전에만 필요
    arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    chans = [sk_rotate(arr[..., c], d, resize=False, preserve_range=True,
                       mode="reflect", order=1) for c in range(3)]
    out = np.clip(np.stack(chans, axis=-1), 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


def round_to_patch_multiple(size: int, patch: int = PATCH_SIZE) -> int:
    """Round ``size`` up to the nearest multiple of ``patch`` (>= patch)."""
    return max(patch, int(round(size / patch)) * patch)


def load_image(path: str) -> Image.Image:
    """Load an image file as an RGB PIL.Image."""
    return Image.open(path).convert("RGB")


def preprocess(img: Image.Image, image_size: int) -> torch.Tensor:
    """Resize + normalize a PIL image into a model-ready tensor.

    Args:
        img: RGB PIL image.
        image_size: target square size (auto-rounded to a multiple of PATCH_SIZE).

    Returns:
        Tensor[3, H, W], float32, ImageNet-normalized.
    """
    size = round_to_patch_multiple(image_size)
    img = img.resize((size, size), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0          # [H, W, 3] in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W]

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Invert ImageNet normalization for visualization.

    Args:
        tensor: Tensor[3, H, W] (normalized).

    Returns:
        np.ndarray[H, W, 3] uint8 in [0, 255].
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = tensor.detach().cpu() * std + mean
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()  # [H, W, 3]
    return (x * 255.0).round().astype(np.uint8)


def load_mask(path: str, image_size: int) -> np.ndarray:
    """Load a binary ground-truth mask, resized to the model input size.

    Args:
        path: mask image path (grayscale, defect=255).
        image_size: target square size (matched to the preprocessed image).

    Returns:
        np.ndarray[H, W] uint8 in {0, 255}. Nearest-neighbor resize to keep it binary.
    """
    size = round_to_patch_multiple(image_size)
    mask = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    arr = np.asarray(mask, dtype=np.uint8)
    return (arr > 127).astype(np.uint8) * 255


def image_hw(tensor: torch.Tensor) -> Tuple[int, int]:
    """Return (H, W) of a [3, H, W] or [B, 3, H, W] tensor."""
    return int(tensor.shape[-2]), int(tensor.shape[-1])
