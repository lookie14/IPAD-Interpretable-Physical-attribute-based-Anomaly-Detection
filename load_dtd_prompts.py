"""
coop_dtd_best_ctx.pt 체크포인트 단독 사용 스크립트
====================================================

train_coop_dtd.py나 다른 파일 없이, 이 체크포인트 하나 + CLIP 라이브러리만
있으면 텍스처 속성 임베딩을 복원해서 이미지와의 유사도를 잴 수 있다.

필요 패키지:
    pip install torch torchvision ftfy regex pillow
    pip install git+https://github.com/openai/CLIP.git

사용법:
    python load_dtd_prompts.py --checkpoint coop_dtd_best_ctx.pt --image my_photo.jpg
"""

import argparse
import re

import clip
import torch
import torch.nn.functional as F
from PIL import Image


def _normalize_backbone_name(name: str) -> str:
    """openai/CLIP wants e.g. "ViT-B/16"; some training configs record it
    filesystem-safe as "ViT-B-16" (no slash). Converts the latter to the
    former; anything already containing "/" (or "RN50" etc.) passes through.
    """
    if "/" in name:
        return name
    m = re.match(r"^(ViT-[A-Z])-(\d+)$", name)
    return f"{m.group(1)}/{m.group(2)}" if m else name


def extract_ctx_fields(ckpt: dict):
    """Pull (ctx, n_ctx, backbone, classnames) out of any supported CoOp
    checkpoint layout:
      - coop_dtd_best_ctx.pt: top-level "ctx" (47 DTD texture attributes)
      - coop_dtd_physical_best.pt: "ctx" nested under "prompt_learner_state"
        (14 physical-defect attributes)
      - coop_dtd_best_ctx_3.pt: a from-scratch training script's own layout --
        "learnable_context"/"context_tokens" instead of "ctx"/"n_ctx", and
        the backbone name nested at config["model"]["name"] in dash form
        (e.g. "ViT-B-16") instead of top-level slash form.
    """
    if "ctx" in ckpt:
        ctx = ckpt["ctx"]
        n_ctx = ckpt["n_ctx"]
        backbone = ckpt["backbone"]
        classnames = ckpt["classnames"]
    elif "prompt_learner_state" in ckpt:
        ctx = ckpt["prompt_learner_state"]["ctx"]
        n_ctx = ckpt["n_ctx"]
        backbone = ckpt["backbone"]
        classnames = ckpt["classnames"]
    elif "learnable_context" in ckpt:
        ctx = ckpt["learnable_context"]
        n_ctx = ckpt["context_tokens"]
        backbone = _normalize_backbone_name(ckpt["config"]["model"]["name"])
        classnames = ckpt["class_names"]
    else:
        raise KeyError(
            "Unrecognized CoOp checkpoint layout -- expected one of top-level "
            "'ctx', 'prompt_learner_state', or 'learnable_context', got keys: "
            f"{list(ckpt.keys())}"
        )
    return ctx, n_ctx, backbone, classnames


def rebuild_text_embeddings(ckpt_path: str, device: str = "cpu"):
    """
    체크포인트에 저장된 컨텍스트 벡터(ctx)로 텍스트 클래스의
    최종 텍스트 임베딩을 다시 계산한다. 학습 때 쓰던 PromptLearner
    클래스가 없어도 되도록, 필요한 최소 로직만 여기 그대로 옮겨왔다.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    ctx, n_ctx, backbone, classnames = extract_ctx_fields(ckpt)
    ctx = ctx.to(device)                     # (n_ctx, ctx_dim)

    clip_model, _ = clip.load(backbone, device=device, jit=False)
    clip_model = clip_model.float().eval()

    # 학습 때와 똑같은 형식으로 프롬프트를 재구성:
    # "[V1]...[Vn] {classname}."
    names = [name.replace("_", " ") for name in classnames]
    prompts = [" ".join(["X"] * n_ctx) + " " + name + "." for name in names]
    tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    with torch.no_grad():
        token_embed = clip_model.token_embedding(tokenized).type(clip_model.dtype)

    prefix = token_embed[:, :1, :]           # SOS 토큰
    suffix = token_embed[:, 1 + n_ctx:, :]   # 클래스명 + EOS 토큰 (고정)
    ctx_expanded = ctx.unsqueeze(0).expand(len(classnames), -1, -1)
    full_prompts = torch.cat([prefix, ctx_expanded, suffix], dim=1)

    with torch.no_grad():
        x = full_prompts + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).type(clip_model.dtype)
        x = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ clip_model.text_projection

    text_features = F.normalize(x, dim=-1)  # (47, embed_dim)
    bank = {name: emb for name, emb in zip(classnames, text_features)}
    return bank, clip_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="coop_dtd_best_ctx.pt")
    parser.add_argument("--image", required=True, help="유사도를 잴 이미지 경로")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bank, clip_model = rebuild_text_embeddings(args.checkpoint, device=device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    _, _, backbone, _ = extract_ctx_fields(ckpt)
    _, preprocess = clip.load(backbone, device=device)

    image = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = F.normalize(clip_model.encode_image(image), dim=-1).squeeze(0)

    sims = {name: (image_feature @ emb).item() for name, emb in bank.items()}
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)

    print(f"'{args.image}'와 텍스처 속성 유사도 상위 {args.topk}개:")
    for name, score in ranked[: args.topk]:
        print(f"  {name:20s} {score:.4f}")


if __name__ == "__main__":
    main()
