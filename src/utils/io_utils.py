"""Config / device / filesystem I/O helpers."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply CLI overrides of the form 'a.b.c=value' onto a config dict (in place).

    Values are parsed as YAML scalars so that ``device=cpu`` -> str,
    ``backbone.layer_index=6`` -> int, ``freeze=false`` -> bool, etc.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def rotation_enabled_for(rot_cfg: Dict[str, Any], category: str) -> bool:
    """이 카테고리에서 회전 정렬을 켤지 결정.

    - rotation.enabled=False 면 항상 off.
    - rotation.enabled_categories 리스트가 있으면 그 목록에 든 카테고리만 on
      (나머지는 off). 리스트가 없거나 비어 있으면 enabled 값을 그대로 사용해
      전 카테고리 동일하게 동작한다.
    """
    rot_cfg = rot_cfg or {}
    if not bool(rot_cfg.get("enabled", False)):
        return False
    cats = rot_cfg.get("enabled_categories")
    if cats:
        return category in set(cats)
    return True


def score_threshold_for(candidate_cfg: Dict[str, Any], category: str,
                        default: float = 0.3) -> float:
    """candidate score_threshold 를 카테고리별로 해석.

    candidate.score_threshold_per_category 맵에 해당 카테고리가 있으면 그 값을,
    없으면 candidate.score_threshold(전역 기본)를 쓴다.
    """
    cc = candidate_cfg or {}
    per = cc.get("score_threshold_per_category") or {}
    if category in per:
        return float(per[category])
    return float(cc.get("score_threshold", default))


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def resolve_device(name: str) -> torch.device:
    """Resolve a requested device string with graceful fallback.

    Order of preference when the request is unavailable: cuda -> mps -> cpu.
    """
    name = (name or "cpu").lower()
    if name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(name)
        if torch.backends.mps.is_available():
            print(f"[device] '{name}' unavailable -> falling back to 'mps'")
            return torch.device("mps")
        print(f"[device] '{name}' unavailable -> falling back to 'cpu'")
        return torch.device("cpu")
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[device] 'mps' unavailable -> falling back to 'cpu'")
        return torch.device("cpu")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Filesystem
# --------------------------------------------------------------------------- #
def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if missing; return it."""
    os.makedirs(path, exist_ok=True)
    return path


def category_output_dirs(
    output_dir: str, category: str, tag: str = "", run_name: str = ""
) -> Dict[str, str]:
    """Build & create the standard Stage 1 output subdirectories for a category.

    ``tag`` (e.g. 'layer06') adds a nested subfolder so multiple runs
    (different layers) do not overwrite each other.
    """
    root = os.path.join(output_dir, category, run_name, tag) if tag else os.path.join(output_dir, category, run_name)
    dirs = {
        "root": root,
        "score_maps": os.path.join(root, "score_maps"),
        "heatmaps": os.path.join(root, "heatmaps"),
        "overlays": os.path.join(root, "overlays"),
        "candidate_boxes": os.path.join(root, "candidate_boxes"),
        "candidate_panels": os.path.join(root, "candidate_panels"),
        "candidates": os.path.join(root, "candidates"),
        "panels": os.path.join(root, "panels"),
    }
    for d in dirs.values():
        ensure_dir(d)
    return dirs


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_pt(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(obj, path)


def save_npy(arr: np.ndarray, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)
