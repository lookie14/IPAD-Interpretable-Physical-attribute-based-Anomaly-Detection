"""Optional Stage 2 add-on: CoOp-learned DTD texture-attribute classifier.

Reuses ``rebuild_text_embeddings`` from the project-root ``load_dtd_prompts.py``
to score each candidate crop against the 47 DTD texture attributes (bumpy,
porous, cracked, ...) that a CoOp checkpoint learned soft prompts for. This is
a separate taxonomy from the defect-type prompts in ``prompt.json`` -- it runs
fully independently (own OpenAI-``clip`` image encoder, own text embeddings)
and is only wired in when ``texture_bank.enabled`` is set in the Stage 2 config.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from load_dtd_prompts import extract_ctx_fields, rebuild_text_embeddings  # noqa: E402


class TextureAttributeClassifier:
    """Scores crops against CoOp-learned DTD texture-attribute prompts.

    Uses the OpenAI ``clip`` library (not HF ``transformers``), since that is
    what the CoOp checkpoint's context vectors were trained against. Image and
    text embeddings both come from the checkpoint's own recorded backbone, so
    this classifier is self-contained and independent of Stage 2's main
    CLIP/SigLIP backend.
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        import clip  # local import: only needed when texture_bank is enabled

        self.device = device
        bank, clip_model = rebuild_text_embeddings(checkpoint_path, device=str(device))
        ckpt = torch.load(checkpoint_path, map_location=device)
        ctx, n_ctx, backbone, _ = extract_ctx_fields(ckpt)
        self.ctx = ctx.to(device)      # (n_ctx, ctx_dim) -- shared across all classnames
        self.n_ctx = n_ctx

        self.labels = list(bank.keys())
        self.text_features = torch.stack([bank[name] for name in self.labels]).to(device)
        self.embedding_bank = dict(zip(self.labels, self.text_features))
        self.embed_dim = self.text_features.shape[-1]
        self.clip_model = clip_model
        _, self.preprocess = clip.load(backbone, device=device, jit=False)
        self.logit_scale = float(clip_model.logit_scale.exp().item())

    def embedding_for(self, label: str) -> torch.Tensor:
        """Return the CoOp checkpoint's own (already L2-normalized) learned text
        embedding for ``label`` -- the same continuous vector used for cosine
        scoring here, as opposed to the discrete label word. Used by Stage 2's
        embedding-space prompt fusion so the soft prompt itself (not a
        re-tokenized word) reaches the defect classifier.
        """
        return self.embedding_bank[label]

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """L2-normalized image embedding via this checkpoint's own CLIP image
        tower. Broken out so a crop encoded once here can be reused by both
        :meth:`classify` and :meth:`classify_fused` instead of re-running the
        (identical) image encoder twice on the same crop.
        """
        pixel_values = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        return F.normalize(self.clip_model.encode_image(pixel_values), dim=-1).squeeze(0)

    @torch.no_grad()
    def classify(
        self, image: Image.Image, top_k: int = 5, img_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Score ``image`` against every DTD texture-attribute prompt.

        ``img_feat``: optional precomputed embedding (e.g. from
        :meth:`encode_image`) to avoid re-encoding the same crop when also
        calling :meth:`classify_fused` on it.
        """
        if img_feat is None:
            img_feat = self.encode_image(image)

        cosine = self.text_features @ img_feat
        probs = F.softmax(cosine * self.logit_scale, dim=0)

        order = torch.argsort(cosine, descending=True).tolist()
        k = min(top_k, len(order))
        results = [
            {
                "label": self.labels[i],
                "cosine": float(cosine[i]),
                "prob": float(probs[i]),
            }
            for i in order[:k]
        ]
        preview = np.asarray(image.resize((224, 224)).convert("RGB"), dtype=np.uint8)
        return results, preview

    @torch.no_grad()
    def encode_with_context(self, texts: List[str]) -> torch.Tensor:
        """Splice this checkpoint's learned ``ctx`` vectors into ``texts`` at
        the token-embedding level -- before the frozen text transformer's
        self-attention layers run -- exactly as ``rebuild_text_embeddings``
        splices ``ctx`` before its own DTD classnames, just with an arbitrary
        piece of text (e.g. a defect prompt) standing in for the classname.
        Self-attention then mixes the learned context with ``texts``' own
        tokens; this is the true CoOp mechanism, as opposed to summing two
        embeddings that were each already fully encoded independently.
        """
        import clip  # local import: only needed when texture_bank is enabled

        placeholder = " ".join(["X"] * self.n_ctx)
        prompts = [f"{placeholder} {t}" for t in texts]
        tokenized = torch.cat([clip.tokenize(p, truncate=True) for p in prompts]).to(self.device)

        token_embed = self.clip_model.token_embedding(tokenized).type(self.clip_model.dtype)
        prefix = token_embed[:, :1, :]                      # SOS token
        suffix = token_embed[:, 1 + self.n_ctx:, :]          # text tokens + EOS + padding (fixed)
        ctx_expanded = self.ctx.unsqueeze(0).expand(len(texts), -1, -1).type(self.clip_model.dtype)
        full_prompts = torch.cat([prefix, ctx_expanded, suffix], dim=1)

        x = full_prompts + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        x = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def classify_fused(
        self,
        image: Image.Image,
        texts: List[str],
        labels: List[str],
        top_k: int = 5,
        img_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Score ``image`` against ``texts`` after fusing each one with this
        checkpoint's learned ctx vectors via :meth:`encode_with_context`.
        Both text and image go through this checkpoint's own CLIP tower
        end-to-end (not Stage 2's main clip.backend) -- the fused embedding
        only lives in that space, so the image side must match.

        ``img_feat``: optional precomputed embedding (e.g. from
        :meth:`encode_image`, or already computed by an earlier
        :meth:`classify` call on the same crop) to avoid a redundant second
        image encode.
        """
        text_features = self.encode_with_context(texts)
        if img_feat is None:
            img_feat = self.encode_image(image)

        cosine = text_features @ img_feat
        probs = F.softmax(cosine * self.logit_scale, dim=0)

        order = torch.argsort(cosine, descending=True).tolist()
        k = min(top_k, len(order))
        results = [
            {
                "label": labels[i],
                "prompt": texts[i],
                "cosine": float(cosine[i]),
                "prob": float(probs[i]),
            }
            for i in order[:k]
        ]
        preview = np.asarray(image.resize((224, 224)).convert("RGB"), dtype=np.uint8)
        return results, preview
