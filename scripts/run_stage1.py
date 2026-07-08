"""Stage 1 runner: aligned multi-layer position-wise anomaly localization.

Usage:
    python3 scripts/run_stage1.py --config configs/stage1_default.yaml \
        --override category=grid device=mps
    python3 scripts/run_stage1.py --config configs/stage1_default.yaml \
        --make-synthetic --override backbone.source=dummy device=cpu

Flow:
    1) load config           2) set seed              3) resolve device
    4) build extractor        5) align+extract refs     6) save ref features
    7) iterate test imgs      8) (optional) derotate    9) extract+score
   10) save heatmap/overlay  11) Top-K/threshold        12) save candidates -> Stage2
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from PIL import Image, ImageDraw

from src.candidate.topk_selector import select_by_threshold, select_topk
from src.datasets.mvtec_dataset import (
    ground_truth_path,
    list_test_images,
    make_synthetic_dataset,
    select_reference_paths,
)
from src.models.dinov3_feature_extractor import DINOv3MultiLayerExtractor
from src.scoring.position_wise_multi_layer_score import resolve_fuse, score as score_fn
from src.utils.image_utils import denormalize, image_hw, load_image, load_mask, preprocess
from src.utils.io_utils import (
    apply_overrides,
    category_output_dirs,
    load_config,
    resolve_device,
    rotation_enabled_for,
    score_threshold_for,
    save_json,
    save_npy,
    save_pt,
)
from src.utils.rotation_adapter import align_reference_set, derotate_pil
from src.utils.seed import set_seed
from src.utils.visualize import (
    save_candidate_boxes,
    save_candidate_panel,
    save_comparison_panel,
    save_heatmap,
    save_overlay,
    upsample_score_map,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 aligned multi-layer position-wise")
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument(
        "--make-synthetic", action="store_true",
        help="generate a tiny synthetic dataset (for dummy smoke test)",
    )
    return p.parse_args()


def extract(extractor, pil, image_size: int, device: torch.device):
    t = preprocess(pil, image_size).unsqueeze(0).to(device)
    feats = extractor(t)
    return {li: f[0] for li, f in feats.items()}, t[0]


def build_aligned_references(cfg: dict, category: str, extractor, device):
    data_root = cfg["data_root"]
    image_size = int(cfg["image_size"])
    ref_paths = select_reference_paths(
        data_root, category, int(cfg["normal_shots"]), seed=int(cfg.get("seed", 42)),
        indices=cfg.get("reference_indices"),
    )
    rot_cfg = cfg.get("rotation") or {}
    rot_on = rotation_enabled_for(rot_cfg, category)
    aligner = None
    if rot_on:
        aligner, ref_pils = align_reference_set(
            ref_paths,
            size=int(rot_cfg.get("size", 256)),
            method=rot_cfg.get("method", "direct"),
            refine=bool(rot_cfg.get("refine", True)),
            background=rot_cfg.get("background", "edge"),
            iters=int(rot_cfg.get("ref_align_iters", 2)),
            center=bool(rot_cfg.get("center", False)),
            tol=float(rot_cfg.get("center_tol", 0.12)),
            matte=bool(rot_cfg.get("matte", False)),
            matte_fill=rot_cfg.get("matte_fill", "auto"),
            matte_model=rot_cfg.get("matte_model", "u2net"),
        )
    else:
        ref_pils = [load_image(p) for p in ref_paths]

    layers = extractor.layer_indices
    per_layer: Dict[int, List[torch.Tensor]] = {li: [] for li in layers}
    for pil in ref_pils:
        feats, _ = extract(extractor, pil, image_size, device)
        for li in layers:
            per_layer[li].append(feats[li].cpu())
    ref_features_cpu = {li: torch.stack(per_layer[li], 0) for li in layers}
    ref_features_dev = {li: ref_features_cpu[li].to(device) for li in layers}
    return ref_features_cpu, ref_features_dev, ref_paths, ref_pils, aligner


def save_reference_debug_images(
    ref_paths: List[str], aligned_pils: List[Image.Image], out_dir: str,
) -> None:
    """Save aligned references and original|aligned comparison images."""
    os.makedirs(out_dir, exist_ok=True)
    for idx, (path, aligned) in enumerate(zip(ref_paths, aligned_pils)):
        base = os.path.splitext(os.path.basename(path))[0]
        aligned_rgb = aligned.convert("RGB")
        aligned_rgb.save(os.path.join(out_dir, f"ref_{idx:03d}_{base}_aligned.png"))

        orig = load_image(path).resize(aligned_rgb.size).convert("RGB")
        w, h = aligned_rgb.size
        pad = 28
        canvas = Image.new("RGB", (w * 2, h + pad), (255, 255, 255))
        canvas.paste(orig, (0, pad))
        canvas.paste(aligned_rgb, (w, pad))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "original", fill=(0, 0, 0))
        draw.text((w + 8, 6), "aligned reference", fill=(0, 0, 0))
        canvas.save(os.path.join(out_dir, f"ref_{idx:03d}_{base}_compare.png"))


def maybe_align_test(pil, cfg: dict, aligner):
    """test 이미지를 정준 pose 로 정렬 (신뢰 score 낮으면 원본 유지).

    주의: root 의 rotation_adapter 는 GT mask 를 같은 변환으로 되돌리는
    derotate_mask 를 제공하지 않으므로, GT 비교(save_comparison)는 회전이
    적용된 경우 원본 GT 좌표와 다소 어긋날 수 있다. 회전을 끈 카테고리에서는
    문제 없음.
    """
    rot_cfg = cfg.get("rotation") or {}
    if not rot_cfg.get("enabled", False) or aligner is None:
        return pil, {"applied": False, "angle": 0.0, "score": None}

    rotated, angle, rscore = derotate_pil(
        pil,
        aligner,
        size=int(rot_cfg.get("size", 256)),
        background=rot_cfg.get("background", "edge"),
        center=bool(rot_cfg.get("center", False)),
        tol=float(rot_cfg.get("center_tol", 0.12)),
        matte=bool(rot_cfg.get("matte", False)),
        matte_fill=rot_cfg.get("matte_fill", "auto"),
        matte_model=rot_cfg.get("matte_model", "u2net"),
    )
    if rscore >= float(rot_cfg.get("min_score", 0.0)):
        return rotated, {"applied": True, "angle": float(angle), "score": float(rscore)}
    return pil, {"applied": False, "angle": float(angle), "score": float(rscore)}


def score_one_orientation(pil, extractor, image_size, device, ref_dev, layers,
                          neighborhoods, fuse_weights, fuse_norm, metric, k_nn):
    feats, t = extract(extractor, pil, image_size, device)
    score_map, layer_score_maps, _rd, _ni = score_fn(
        feats, ref_dev, layers, neighborhoods,
        weights=fuse_weights, normalize=fuse_norm, metric=metric, k=k_nn,
    )
    feat_map = torch.cat([feats[li].cpu() for li in layers], dim=-1)
    return feats, t, score_map.cpu(), layer_score_maps.cpu(), feat_map


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cpu"))
    image_size = int(cfg["image_size"])

    extractor = DINOv3MultiLayerExtractor(cfg).to(device)
    layers = extractor.layer_indices
    metric = cfg["scoring"].get("distance_metric", "cosine")
    k_nn = int(cfg["scoring"].get("k", 1))
    neighborhoods, fuse_weights, fuse_norm = resolve_fuse(cfg["scoring"])

    category = cfg["category"]
    run_name = cfg.get("run_name") or category
    print(f"[run] stage1 category={category} run_name={run_name} device={device}")
    print(f"[run] layers={layers} metric={metric} k={k_nn} neighborhoods={neighborhoods}")
    print(f"[run] rotation={rotation_enabled_for(cfg.get('rotation') or {}, category)}")

    if args.make_synthetic:
        make_synthetic_dataset(cfg["data_root"], category, image_size=image_size,
                               seed=int(cfg.get("seed", 42)))

    dirs = category_output_dirs(cfg["output_dir"], category, run_name=run_name)
    ref_cpu, ref_dev, ref_paths, aligned_ref_pils, aligner = build_aligned_references(
        cfg, category, extractor, device
    )
    h_p, w_p, c = ref_cpu[layers[0]].shape[1:]
    print(f"[run] refs={len(ref_paths)} grid={h_p}x{w_p} C={c}")
    if cfg["visualization"].get("save_aligned_refs", True):
        save_reference_debug_images(
            ref_paths, aligned_ref_pils, os.path.join(dirs["root"], "aligned_refs"),
        )
        print(f"[run] saved aligned references -> {os.path.join(dirs['root'], 'aligned_refs')}")
    save_pt(
        {
            "ref_features": ref_cpu,
            "ref_image_paths": ref_paths,
            "category": category,
            "backbone": cfg["backbone"]["name"],
            "layer_indices": layers,
            "image_size": image_size,
            "normal_shots": int(cfg["normal_shots"]),
            "scoring": cfg["scoring"],
            "rotation": cfg.get("rotation"),
        },
        os.path.join(dirs["root"], "reference_features.pt"),
    )

    cand_cfg = cfg["candidate"]
    cand_method = cand_cfg.get("method", "topk")
    topk_ratio = float(cand_cfg.get("topk_ratio", 0.05))
    topk_k = cand_cfg.get("topk_k")
    topk_k = int(topk_k) if topk_k is not None else None
    score_threshold = score_threshold_for(cand_cfg, category, 0.3)
    if cand_method == "threshold":
        print(f"[run] score_threshold={score_threshold} (category={category})")
    save_feature = bool(cand_cfg.get("save_feature", True))
    vis = cfg["visualization"]
    alpha = float(vis.get("overlay_alpha", 0.5))

    items = list_test_images(
        cfg["data_root"], category,
        anomaly_types=cfg.get("test_anomaly_types"),
        indices=cfg.get("test_indices"),
    )
    print(f"[run] test images={len(items)} candidate_method={cand_method}")

    for path, anomaly_type in items:
        name = f"{anomaly_type}_{os.path.splitext(os.path.basename(path))[0]}"
        raw = load_image(path)
        pil, align_info = maybe_align_test(raw, cfg, aligner)
        _feats, t, score_map, layer_score_maps, feat_map = score_one_orientation(
            pil, extractor, image_size, device, ref_dev, layers, neighborhoods,
            fuse_weights, fuse_norm, metric, k_nn,
        )

        img_hw = image_hw(t)
        orig = denormalize(t)
        score_up = upsample_score_map(score_map, img_hw)

        if vis.get("save_score_map_npy", True):
            save_npy(score_map.numpy(), os.path.join(dirs["score_maps"], f"{name}.npy"))
        if vis.get("save_heatmap", True):
            save_heatmap(score_up, os.path.join(dirs["heatmaps"], f"{name}.png"))
        if vis.get("save_overlay", True):
            save_overlay(orig, score_up, os.path.join(dirs["overlays"], f"{name}.png"), alpha)
        if vis.get("save_comparison", True):
            gt_path = ground_truth_path(cfg["data_root"], category, anomaly_type, path)
            gt_mask = load_mask(gt_path, image_size) if gt_path else None
            save_comparison_panel(
                orig, gt_mask, score_up, os.path.join(dirs["panels"], f"{name}.png"),
                alpha=alpha, title=f"{category}/{name}",
            )

        if cand_method == "threshold":
            candidates = select_by_threshold(
                score_map, layer_score_maps, img_hw, (h_p, w_p),
                threshold=score_threshold, feature_map=feat_map, save_feature=save_feature,
            )
        else:
            candidates = select_topk(
                score_map, layer_score_maps, img_hw, (h_p, w_p),
                topk_ratio=topk_ratio, topk_k=topk_k,
                feature_map=feat_map, save_feature=save_feature,
            )

        if vis.get("save_candidate_boxes", True):
            save_candidate_boxes(
                orig, candidates, os.path.join(dirs["candidate_boxes"], f"{name}.png"),
            )
        if vis.get("save_candidate_panel", True):
            save_candidate_panel(
                orig, candidates, score_up,
                os.path.join(dirs["candidate_panels"], f"{name}.png"),
                alpha=alpha, title=f"{category}/{name}",
            )

        json_candidates = [{kk: vv for kk, vv in cnd.items() if kk != "feature"} for cnd in candidates]
        save_json(
            {
                "image_path": path,
                "category": category,
                "anomaly_type": anomaly_type,
                "image_size": [img_hw[0], img_hw[1]],
                "patch_grid_size": [h_p, w_p],
                "layer_indices": layers,
                "alignment": align_info,
                "candidates": json_candidates,
            },
            os.path.join(dirs["candidates"], f"{name}.json"),
        )
        save_pt(
            {
                "image_path": path,
                "score_map": score_map,
                "layer_score_maps": layer_score_maps,
                "feature_map": feat_map,
                "alignment": align_info,
                "candidates": candidates,
            },
            os.path.join(dirs["candidates"], f"{name}.pt"),
        )
        msg = f"[run] {name}: {len(candidates)} candidates score[min={score_map.min():.3f}, max={score_map.max():.3f}]"
        if align_info.get("score") is not None:
            msg += f" align[applied={align_info['applied']} angle={align_info['angle']:.1f} score={align_info['score']:.2f}]"
        print(msg)


if __name__ == "__main__":
    main()
