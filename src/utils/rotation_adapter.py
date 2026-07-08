"""Rotation-alignment adapter for the Stage 1 pipeline.

`src/utils/rotation.py` 의 RotationAligner 를 감싼다. 핵심:
  - reference 이미지들로 aligner(정준 pose 템플릿)를 한 번 만든다.
  - test 이미지는 **회전각만 추정**해서 **원본 RGB에 그대로 적용** → grayscale 로
    바꾸지 않아 색 정보(예: leather color 결함)를 잃지 않는다.
  - reference 는 정준 pose 그 자체이므로 회전하지 않는다.

feature 추출(전처리) 직전에 삽입한다.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize as sk_resize
from skimage.transform import rotate as sk_rotate

from src.utils.image_utils import load_image
from src.utils.rotation import RotationAligner


# --------------------------------------------------------------------------- #
# 누끼(전경 분리) — 배경이 복잡할 때 회전 잔상 제거용
#
# 배경이 복잡한 이미지를 회전하면 배경 텍스처까지 같이 돌면서 잔상(ghosting)이
# 남아 patch 단계에서 가짜 이상으로 잡힌다. 회전 전에 전경만 남기고 배경을
# 단색(또는 template)으로 대체하면, 균일한 배경은 회전해도 잔상이 없다.
#
# 엔진: rembg(U2Net). 딥러닝 기반이라 복잡한 배경에도 강함.
#   pip install rembg   (첫 실행 시 모델 자동 다운로드)
# --------------------------------------------------------------------------- #
_REMBG_SESSIONS: dict = {}


def _get_rembg_session(model_name: str = "u2net"):
    """rembg 세션을 (모델별) 캐싱. 모델 로드 비용이 커서 한 번만 만든다."""
    sess = _REMBG_SESSIONS.get(model_name)
    if sess is None:
        try:
            from rembg import new_session
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "누끼(matte) 기능은 rembg 패키지가 필요합니다. "
                "`pip install rembg` 후 다시 실행하세요."
            ) from e
        sess = new_session(model_name)
        _REMBG_SESSIONS[model_name] = sess
    return sess


def _foreground_alpha(rgb: np.ndarray, model_name: str = "u2net") -> np.ndarray:
    """rembg 로 전경 alpha 맵을 추정.

    Args:
        rgb: [H, W, 3] float in [0, 1].
        model_name: rembg 모델 이름 (u2net, u2netp, isnet-general-use 등).

    Returns:
        [H, W] float alpha in [0, 1] (전경=1, 배경=0). 소프트 매트라 가장자리가
        부드럽다.
    """
    from rembg import remove

    session = _get_rembg_session(model_name)
    src = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    pil_in = Image.fromarray(src, mode="RGB")
    cut = remove(pil_in, session=session)              # RGBA
    alpha = np.asarray(cut.convert("RGBA"), dtype=np.float64)[..., 3] / 255.0
    return alpha


def _resolve_fill(
    rgb: np.ndarray, alpha: np.ndarray, fill: "str | float | tuple | list"
) -> np.ndarray:
    """배경을 채울 색(길이-3 RGB, [0,1])을 결정.

    fill:
      "auto"   -> 배경(alpha<0.5) 픽셀들의 채널별 median (없으면 테두리 median).
      "white"  -> (1, 1, 1)
      "black"  -> (0, 0, 0)
      <float>  -> 회색 상수 (모든 채널 동일)
      (r,g,b)  -> 지정한 RGB 상수 ([0,1] 또는 0-255 자동 판별).
    """
    if isinstance(fill, str):
        key = fill.lower()
        if key == "white":
            return np.ones(3, dtype=np.float64)
        if key == "black":
            return np.zeros(3, dtype=np.float64)
        if key == "auto":
            bg = rgb[alpha < 0.5]
            if bg.shape[0] >= 30:
                return np.median(bg.reshape(-1, 3), axis=0)
            b = max(2, min(rgb.shape[:2]) // 32)        # 테두리 프레임 fallback
            frame = np.concatenate([
                rgb[:b, :, :].reshape(-1, 3), rgb[-b:, :, :].reshape(-1, 3),
                rgb[:, :b, :].reshape(-1, 3), rgb[:, -b:, :].reshape(-1, 3),
            ])
            return np.median(frame, axis=0)
        raise ValueError(f"알 수 없는 fill 문자열: {fill!r}")
    if isinstance(fill, (tuple, list, np.ndarray)):
        c = np.asarray(fill, dtype=np.float64).reshape(-1)
        if c.size != 3:
            raise ValueError("RGB fill 은 길이 3 이어야 합니다.")
        if c.max() > 1.0:                               # 0-255 입력 자동 정규화
            c = c / 255.0
        return c
    return np.full(3, float(fill), dtype=np.float64)    # 회색 상수


def remove_background_rgb(
    rgb: np.ndarray,
    fill: "str | float | tuple | list" = "auto",
    model_name: str = "u2net",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """전경 누끼를 따서 배경을 단색으로 대체.

    Args:
        rgb: [H, W, 3] float in [0, 1].
        fill: 배경 채움 색 (_resolve_fill 참고).
        model_name: rembg 모델.

    Returns:
        (out, alpha, fill_rgb)
        - out: [H, W, 3] 배경이 단색으로 바뀐 이미지 (소프트 알파 합성).
        - alpha: [H, W] 전경 알파 (0~1).
        - fill_rgb: 사용된 배경 색 (길이-3, [0,1]) — 회전 모서리 채움에 재사용.
    """
    alpha = _foreground_alpha(rgb, model_name)
    fill_rgb = _resolve_fill(rgb, alpha, fill)
    a = alpha[..., None]
    out = rgb * a + fill_rgb[None, None, :] * (1.0 - a)
    return np.clip(out, 0.0, 1.0), alpha, fill_rgb


def _matte_pil(
    pil_img: Image.Image,
    fill: "str | float | tuple | list" = "auto",
    model_name: str = "u2net",
) -> Image.Image:
    """PIL RGB → 누끼 처리된 PIL RGB (배경 단색)."""
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float64) / 255.0
    out, _alpha, _fill = remove_background_rgb(rgb, fill=fill, model_name=model_name)
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


def _center_rgb(rgb: np.ndarray, tol: float = 0.12) -> np.ndarray:
    """전경 물체의 무게중심을 이미지 중앙으로 이동 (translation 보정).

    회전은 이미지 중심 기준이라, 물체가 프레임 안에서 이동하면(특히 'direct'
    방식) 각도 추정이 흔들린다. 회전 정렬 전에 위치를 맞춰준다.

    Args:
        rgb: [H, W, 3] float in [0, 1].
        tol: 배경값과 이 이상 차이나는 픽셀을 전경으로 간주.

    Returns:
        [H, W, 3] 위치 보정된 이미지 (모서리는 edge 연장).
    """
    from scipy.ndimage import center_of_mass
    from scipy.ndimage import shift as nd_shift

    gray = rgb2gray(rgb)
    b = max(2, min(gray.shape) // 32)
    border = np.concatenate([
        gray[:b, :].ravel(), gray[-b:, :].ravel(),
        gray[:, :b].ravel(), gray[:, -b:].ravel(),
    ])
    bg = float(np.median(border))
    fg = np.abs(gray - bg) > tol
    if fg.sum() < 10:                       # 전경이 불분명하면 그대로
        return rgb
    cy, cx = center_of_mass(fg)
    h, w = gray.shape
    dy, dx = (h - 1) / 2.0 - cy, (w - 1) / 2.0 - cx
    return np.stack(
        [nd_shift(rgb[..., c], (dy, dx), mode="nearest") for c in range(3)], axis=-1)


def _center_pil(pil_img: Image.Image, tol: float = 0.12) -> Image.Image:
    """PIL RGB → 위치 보정된 PIL RGB."""
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float64) / 255.0
    out = np.clip(_center_rgb(rgb, tol), 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


def _to_size_rgb(img: "Image.Image | np.ndarray", size: int) -> np.ndarray:
    """PIL/array → [size, size, 3] float64 in [0, 1]."""
    rgb = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img,
                     dtype=np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    return sk_resize(rgb, (size, size), preserve_range=True, anti_aliasing=True)


def build_aligner_from_images(
    images: List["Image.Image | np.ndarray"],
    size: int = 256,
    method: str = "fourier",
    refine: bool = True,
) -> RotationAligner:
    """이미지(PIL/array) 리스트로 RotationAligner 생성 (내부에서 grayscale 변환)."""
    refs = [_to_size_rgb(im, size) for im in images]
    return RotationAligner(refs, method=method, refine=refine)


def build_aligner(
    ref_paths: List[str],
    size: int = 256,
    method: str = "fourier",
    refine: bool = True,
) -> RotationAligner:
    """정상 reference 경로들로 RotationAligner 를 만든다 (정준 pose 가정)."""
    return build_aligner_from_images(
        [load_image(p) for p in ref_paths], size, method, refine)


def align_reference_set(
    ref_paths: List[str],
    size: int = 256,
    method: str = "fourier",
    refine: bool = True,
    background: "str | float" = "edge",
    iters: int = 2,
    center: bool = False,
    tol: float = 0.12,
    matte: bool = False,
    matte_fill: "str | float | tuple | list" = "auto",
    matte_model: str = "u2net",
) -> Tuple[RotationAligner, List[Image.Image]]:
    """reference 이미지들을 서로 **공통 pose 로 정렬**한다.

    reference 끼리 회전이 제각각인 경우(그냥 평균 내면 템플릿이 흐려짐)를 위한 처리.
      1) ref[0] 를 초기 템플릿으로 나머지를 정렬 (부트스트랩).
      2) 정렬된 집합의 평균으로 템플릿을 다시 만들고 전체를 재정렬 (iters-1 회 반복).
      3) 최종 정렬 집합으로 aligner 를 만들어 반환.

    matte=True 면 reference 도 회전 정렬 전에 누끼를 따서 배경을 단색으로 만든다.
    test 도 같은 옵션으로 처리해야 template 과 pose·배경이 일치한다.

    Returns:
        (final_aligner, aligned_pils)
        - final_aligner: 정렬된 정상 집합으로 만든 aligner (test 는 여기에 맞춰 정렬)
        - aligned_pils: 공통 pose 로 회전된 reference RGB PIL 리스트
          → 이걸로 reference feature 를 뽑아야 test 와 pose 가 일치함.
    """
    pil_refs = [load_image(p) for p in ref_paths]
    if matte:                                        # 누끼 먼저 (배경 단색화)
        pil_refs = [_matte_pil(p, matte_fill, matte_model) for p in pil_refs]
    if center:                                       # 위치 보정 (한 번만)
        pil_refs = [_center_pil(p, tol) for p in pil_refs]
    if len(pil_refs) == 1:
        return build_aligner_from_images(pil_refs, size, method, refine), pil_refs

    # 부트스트랩: ref[0] 를 템플릿으로 (이미 centered → 아래 회전 정렬은 center=False)
    aligner = build_aligner_from_images([pil_refs[0]], size, method, refine)
    aligned = [pil_refs[0]] + [
        derotate_pil(p, aligner, size, background)[0] for p in pil_refs[1:]
    ]

    # 템플릿 재추정 + 전체 재정렬 반복
    for _ in range(max(0, iters - 1)):
        aligner = build_aligner_from_images(aligned, size, method, refine)
        aligned = [derotate_pil(p, aligner, size, background)[0] for p in pil_refs]

    final_aligner = build_aligner_from_images(aligned, size, method, refine)
    return final_aligner, aligned


def derotate_pil(
    pil_img: Image.Image,
    aligner: RotationAligner,
    size: int = 256,
    background: "str | float" = "edge",
    center: bool = False,
    tol: float = 0.12,
    matte: bool = False,
    matte_fill: "str | float | tuple | list" = "auto",
    matte_model: str = "u2net",
) -> Tuple[Image.Image, float, float]:
    """test 이미지를 정준 pose 로 회전시킨 RGB PIL 을 반환.

    회전각은 (size x size) grayscale 로 추정하고, 회전은 **원본 해상도 RGB** 에
    채널별로 적용한다(각도는 해상도 불변).

    Args:
        pil_img: 입력 RGB PIL.
        aligner: build_aligner 로 만든 aligner.
        size: 각도 추정 해상도 (aligner 템플릿과 동일해야 함).
        background: 회전으로 드러난 모서리 채우기. "edge"(테두리 연장) 또는 상수값.
        center: True 면 회전 전에 물체 위치를 중앙으로 보정 (aligner 도 centered
                reference 로 만들어야 pose 가 일치함).
        tol: centering 전경 판정 임계값.
        matte: True 면 회전 **전에** 누끼(전경 분리)를 따서 배경을 단색으로
               대체한다. 배경이 복잡해 회전 잔상이 심할 때 켠다. 이때 회전
               모서리도 같은 배경색으로 채워 전체가 균일해진다.
        matte_fill: 누끼 후 배경 채움 색 (remove_background_rgb 참고).
        matte_model: rembg 모델 이름.

    Returns:
        (회전된 RGB PIL, angle_deg, score).  score 는 정렬 신뢰도(-1~1).
    """
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float64) / 255.0   # [H, W, 3]

    fill_rgb = None
    if matte:                                                           # 누끼 먼저
        rgb, _alpha, fill_rgb = remove_background_rgb(
            rgb, fill=matte_fill, model_name=matte_model)
        # 배경이 단색이 됐으니 회전 모서리도 같은 색으로 채워 잔상/경계선 방지.
        if background == "edge":
            background = fill_rgb

    if center:
        rgb = _center_rgb(rgb, tol)                                     # translation 보정

    # 각도 추정 (grayscale 변환은 aligner 내부에서 처리)
    small = sk_resize(rgb, (size, size), preserve_range=True, anti_aliasing=True)
    res = aligner.align(small)
    angle, score = float(res.angle), float(res.score)

    # 원본 RGB 를 -angle 만큼 회전 (aligner 규약: canonical = rotate(img, -angle))
    out = _rotate_rgb(rgb, -angle, background)
    pil_out = Image.fromarray((out * 255.0).round().astype(np.uint8))
    return pil_out, angle, score


def _rotate_rgb(
    rgb: np.ndarray, angle: float, background: "str | float | tuple | list | np.ndarray"
) -> np.ndarray:
    """[H, W, 3] RGB 를 angle(deg) 만큼 회전. 모서리 채움 방식은 background 로.

    background:
      "edge"            -> 테두리 픽셀 연장 (skimage mode='edge').
      <float>           -> 모든 채널 동일 회색 상수.
      (r, g, b)/ndarray -> 채널별 상수 (누끼 배경색 재사용).
    """
    kw = dict(resize=False, preserve_range=True)
    if isinstance(background, str) and background == "edge":
        kw["mode"] = "edge"
        chans = [sk_rotate(rgb[..., c], angle, **kw) for c in range(3)]
    elif isinstance(background, (tuple, list, np.ndarray)):
        cval = np.asarray(background, dtype=np.float64).reshape(-1)
        kw["mode"] = "constant"
        chans = [sk_rotate(rgb[..., c], angle, cval=float(cval[c]), **kw)
                 for c in range(3)]
    else:                                                              # 회색 상수
        kw["mode"] = "constant"
        kw["cval"] = float(background)
        chans = [sk_rotate(rgb[..., c], angle, **kw) for c in range(3)]
    return np.clip(np.stack(chans, axis=-1), 0.0, 1.0)
