"""Stage 2 runner: CoOp 학습된 physical-property 프롬프트로 crop 분류.

Usage:
    python scripts/run_stage2.py --config configs/stage2_default.yaml
    python scripts/run_stage2.py --config configs/stage2_default.yaml \
        --override category=leather device=cpu

Flow (per Stage 1 image) — design spec(1.png) 그대로, 클러스터링/시각적 강조 없음:
    1) Stage 1 candidate(patch) 각각을 독립된 region 으로 처리
    2) candidate bbox 에 padding 을 더해 crop window 생성 후 원본(Stage-1-resized)
       이미지를 crop
    3) CoOp 학습된 CLIP(coop_dtd_best_ctx_3.pt)으로 crop 을 physical-property
       클래스와 비교 -> Top-K
    4) 모든 candidate 를 TypedRegion 으로 저장
       {bbox, patch_coords, geo_score, pred_attr, attr_probs, sem_score}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PIL import Image

from src.pipeline.stage2_core import build_classifier, process_candidates
from src.utils.image_utils import denormalize, load_image, preprocess
from src.utils.io_utils import apply_overrides, ensure_dir, load_config, resolve_device
from src.utils.seed import set_seed
from src.utils.visualize import save_image, save_topk_bar


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 CoOp physical-property classification")
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument(
        "--override", nargs="*", default=[],
        help="config overrides, e.g. category=leather device=cpu",
    )
    return p.parse_args()


def stage1_candidate_files(stage1_dir: str, category: str, tag: str) -> list:
    root = os.path.join(stage1_dir, category, tag) if tag else os.path.join(stage1_dir, category)
    cand_dir = os.path.join(root, "candidates")
    if not os.path.isdir(cand_dir):
        raise FileNotFoundError(
            f"[s2] No Stage 1 candidates dir at '{cand_dir}'. "
            "Run scripts/run_stage1.py first (with a matching category/tag)."
        )
    files = sorted(glob.glob(os.path.join(cand_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"[s2] No candidate JSON files found in '{cand_dir}'.")
    return files


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cpu"))
    print(f"[s2] device = {device}")

    category = cfg["category"]
    stage1_dir = cfg["stage1_dir"]
    stage1_tag = cfg.get("stage1_tag", "")
    output_dir = cfg["output_dir"]
    crop_cfg = cfg["crop"]
    vis = cfg.get("visualization", {})
    top_k = int(cfg.get("topk", 5))
    sem_score_threshold = float(cfg.get("sem_score_threshold", 0.0))

    classifier = build_classifier(cfg["checkpoint"], device)
    print(f"[s2] CoOp checkpoint = {cfg['checkpoint']} "
          f"({len(classifier.labels)} physical-property classes: {', '.join(classifier.labels)})")

    candidate_files = stage1_candidate_files(stage1_dir, category, stage1_tag)
    print(f"[s2] {len(candidate_files)} Stage 1 images with candidates")

    out_root = os.path.join(output_dir, category, stage1_tag) if stage1_tag else os.path.join(output_dir, category)
    ensure_dir(out_root)

    for json_path in candidate_files:
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)

        name = os.path.splitext(os.path.basename(json_path))[0]
        image_path = record["image_path"]
        image_hw = tuple(record["image_size"])   # (H, W), square (Stage 1 preprocess)
        candidates = record["candidates"]

        t = preprocess(load_image(image_path), image_hw[0])
        orig_image = Image.fromarray(denormalize(t), "RGB")

        regions = process_candidates(
            orig_image, candidates, image_hw, classifier,
            padding_ratio=float(crop_cfg.get("padding_ratio", 1.0)),
            padding_px=crop_cfg.get("padding_px"),
            min_crop_size=int(crop_cfg.get("min_crop_size", 64)),
            square=bool(crop_cfg.get("square", True)),
            top_k=top_k,
            sem_score_threshold=sem_score_threshold,
        )
        print(f"[s2] {name}: {len(candidates)} Stage-1 patches -> {len(regions)} regions "
              f"(sem_score_threshold={sem_score_threshold})")

        image_dir = os.path.join(out_root, name)
        ensure_dir(image_dir)
        json_regions = []
        for idx, r in enumerate(regions):
            region_dir = os.path.join(image_dir, f"region_{idx:03d}")
            ensure_dir(region_dir)
            if vis.get("save_crop", True):
                save_image(
                    r["padded_crop"], os.path.join(region_dir, "crop.png"),
                    title=f"{r['pred_attr']} ({r['sem_score']:.2f})",
                )
            if vis.get("save_topk_bar", True):
                save_topk_bar(
                    r["topk"], os.path.join(region_dir, "topk_bar.png"),
                    title="Top-K physical-property match",
                )
            json_regions.append({k: v for k, v in r.items() if k != "padded_crop"})

        with open(os.path.join(image_dir, "regions.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"image_path": image_path, "category": category, "regions": json_regions},
                f, indent=2, ensure_ascii=False,
            )
        if regions:
            top = regions[0]
            print(f"[s2] {name}: top region patch_coords={top['patch_coords']} "
                  f"pred_attr={top['pred_attr']} sem_score={top['sem_score']:.3f}")
        else:
            print(f"[s2] {name}: no Stage-1 candidates -- no regions")

    print(f"[s2] done -> {out_root}")


if __name__ == "__main__":
    main()
