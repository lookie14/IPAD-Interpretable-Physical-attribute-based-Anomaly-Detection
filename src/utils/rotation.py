"""
rotation_align.py

Global de-rotation preprocessing for patch-based anomaly detection.

Given a small set of aligned "normal" reference images, this module estimates
the in-plane rotation of an incoming test image relative to that reference and
rotates it back to the canonical orientation *before* patch extraction.

Core idea (Fourier-Mellin / log-polar phase correlation):
  - A rotation about the image center becomes a pure translation (a shift along
    the angular axis) in log-polar coordinates.
  - phase_cross_correlation recovers that shift -> the rotation angle.
  - Working on the FFT *magnitude* spectrum makes the angle estimate robust to
    residual translation, at the cost of a 180-degree ambiguity (the magnitude
    spectrum of a real image is centro-symmetric). We resolve that ambiguity by
    de-rotating with each candidate and keeping whichever best matches the
    reference template.

Only a coarse alignment is needed for PatchCore / AnomalyDINO-style pipelines:
patch nearest-neighbour matching absorbs a few degrees of residual error. An
optional fine brute-force refinement (+/- a few degrees) is included for cases
where you want tighter alignment.

Dependencies: numpy, scipy, scikit-image (all standard in Colab).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.fft import fft2, fftshift
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.registration import phase_cross_correlation
from skimage.transform import resize as sk_resize
from skimage.transform import rotate, warp_polar

ANG_BINS = 360  # angular resolution of the polar warp: 1 bin == 1 degree


# --------------------------------------------------------------------------- #
# Quickstart (Colab + Google Drive)
# --------------------------------------------------------------------------- #
#   from google.colab import drive; drive.mount('/content/drive')
#   from rotation_align import RotationAligner, load_images_from_dir
#
#   refs = load_images_from_dir('/content/drive/MyDrive/normal_refs', size=(256, 256))
#   aligner = RotationAligner(refs, method='fourier', refine=True)
#
#   res = aligner.align(test_img)        # test_img: HxW or HxWxC array
#   patches = extract_patches(res.image) # feed res.image (canonical pose) to your
#                                        # PatchCore / AnomalyDINO patch extractor
#   if res.score < 0.5:                  # low similarity -> alignment unreliable
#       ...                              #   (heavy anomaly, wrong category, etc.)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Convert any image to a float64 grayscale array in [0, 1]."""
    img = np.asarray(img)
    if img.ndim == 3:
        if img.shape[2] == 4:  # drop alpha
            img = img[..., :3]
        img = rgb2gray(img)
    img = img.astype(np.float64)
    mx = img.max()
    if mx > 1.0:  # assume 0-255 input
        img = img / 255.0
    return img


def _circular_mask(shape: tuple[int, int], frac: float = 0.95) -> np.ndarray:
    """Boolean disk mask; used to score similarity while ignoring the black
    corners introduced by rotation."""
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx)
    return r <= frac * min(cy, cx)


def _hann2d(shape: tuple[int, int]) -> np.ndarray:
    """Separable 2-D Hann window to suppress FFT edge leakage."""
    h, w = shape
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx)


def _highpass(shape: tuple[int, int]) -> np.ndarray:
    """High-emphasis filter (Reddy & Chatterji) applied to the log-magnitude
    spectrum before the polar warp; downweights the DC-dominated low
    frequencies so the angle estimate keys on structure, not overall energy."""
    h, w = shape
    yy = np.linspace(-0.5, 0.5, h)[:, None]
    xx = np.linspace(-0.5, 0.5, w)[None, :]
    rad = np.cos(np.pi * yy) * np.cos(np.pi * xx)
    return (1.0 - rad) * (2.0 - rad)


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Masked normalized cross-correlation (Pearson) in [-1, 1]."""
    av = a[mask]
    bv = b[mask]
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.sqrt((av * av).sum() * (bv * bv).sum()) + 1e-12
    return float((av * bv).sum() / denom)


def load_images_from_dir(
    directory: str,
    size: tuple[int, int] | None = None,
    exts: tuple[str, ...] = ("png", "jpg", "jpeg", "bmp", "tif", "tiff"),
) -> list[np.ndarray]:
    """Load every image in a folder (handy for Colab + mounted Drive).

    Parameters
    ----------
    directory : path to the folder of aligned normal images.
    size      : optional (H, W) to resize every image to a common shape.
    exts      : file extensions to include (case-insensitive).
    """
    paths = sorted(
        p
        for e in exts
        for p in glob.glob(os.path.join(directory, f"*.{e}"))
        + glob.glob(os.path.join(directory, f"*.{e.upper()}"))
    )
    if not paths:
        raise FileNotFoundError(f"No images with {exts} found in {directory!r}.")
    imgs = []
    for p in paths:
        im = imread(p)
        if size is not None:
            im = sk_resize(im, size, preserve_range=True, anti_aliasing=True)
        imgs.append(im)
    return imgs


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class AlignResult:
    image: np.ndarray      # de-rotated grayscale float image, same shape as input
    angle: float           # rotation (deg) applied to the query to canonicalize it
    score: float           # masked NCC vs. template after alignment, in [-1, 1]
    candidates: list       # (candidate_angle, score) pairs that were evaluated


# --------------------------------------------------------------------------- #
# Aligner
# --------------------------------------------------------------------------- #
class RotationAligner:
    """
    Build once from your aligned normal images, then call `.align(query)` on each
    test image before patch extraction.

    Parameters
    ----------
    references : sequence of images (H, W) or (H, W, C)
        The aligned "normal" images. They are averaged into a single template.
    method : {"fourier", "direct"}
        "fourier"  -> FFT-magnitude log-polar (robust to residual translation,
                      resolves the 180-deg ambiguity by template matching).
        "direct"   -> log-polar directly on the image (simplest; assumes the
                      object is well centered and untranslated).
    refine : bool
        If True, run a fine brute-force sweep (+/- `refine_range` deg) around the
        coarse estimate to squeeze out residual error.
    refine_range, refine_step : float
        Window and resolution (deg) of the refinement sweep.
    """

    def __init__(
        self,
        references: Iterable[np.ndarray],
        method: Literal["fourier", "direct"] = "fourier",
        refine: bool = True,
        refine_range: float = 4.0,
        refine_step: float = 0.5,
        background: "str | float" = "template",
    ) -> None:
        refs = [_to_gray_float(r) for r in references]
        if not refs:
            raise ValueError("Provide at least one reference image.")
        shape = refs[0].shape
        if any(r.shape != shape for r in refs):
            raise ValueError("All reference images must share the same shape.")

        self.method = method
        self.refine = refine
        self.refine_range = refine_range
        self.refine_step = refine_step
        self.background = background  # default fill mode for align(); overridable

        # Averaged template — smooths per-sample noise, keeps the shared pose.
        self.template = np.mean(refs, axis=0)
        self.shape = shape
        self.radius = min(shape) // 2
        self.mask = _circular_mask(shape)
        self._win = _hann2d(shape)
        self._hp = _highpass(shape)

        # Scalar background estimate for background="auto": median of a thin
        # border frame of the template (assumes the object is centered, so the
        # frame is mostly background).
        b = max(2, min(shape) // 32)
        border = np.concatenate(
            [
                self.template[:b, :].ravel(),
                self.template[-b:, :].ravel(),
                self.template[:, :b].ravel(),
                self.template[:, -b:].ravel(),
            ]
        )
        self._bg_value = float(np.median(border))

        # Precompute the template's polar representation for the chosen method.
        self._tmpl_polar = self._polar(self.template)

    # -- polar representations -------------------------------------------- #
    def _polar(self, img: np.ndarray) -> np.ndarray:
        if self.method == "direct":
            return warp_polar(
                img, radius=self.radius, output_shape=(ANG_BINS, self.radius)
            )
        # Fourier-magnitude branch
        spec = np.abs(fftshift(fft2(img * self._win)))
        spec = np.log1p(spec) * self._hp
        # Only 180 deg of angular span is independent (centro-symmetry), but we
        # warp the full 360 and resolve the ambiguity later against the template.
        return warp_polar(
            spec, radius=self.radius, output_shape=(ANG_BINS, self.radius)
        )

    # -- coarse angle ----------------------------------------------------- #
    def _coarse_angle(self, query: np.ndarray) -> float:
        q_polar = self._polar(query)
        shift, _err, _phase = phase_cross_correlation(
            self._tmpl_polar, q_polar, upsample_factor=10, normalization=None
        )
        # shift along the angular axis (bin == degree) is the rotation estimate.
        return float(shift[0]) % 360.0

    # -- similarity of a de-rotated query to the template ----------------- #
    def _score_angle(self, query: np.ndarray, angle: float) -> float:
        rotated = rotate(query, -angle, resize=False, preserve_range=True)
        return _ncc(rotated, self.template, self.mask)

    # -- optional fine refinement ----------------------------------------- #
    def _refine_angle(self, query: np.ndarray, angle: float) -> tuple[float, float]:
        best_a, best_s = angle, self._score_angle(query, angle)
        sweep = np.arange(
            angle - self.refine_range,
            angle + self.refine_range + 1e-9,
            self.refine_step,
        )
        for a in sweep:
            s = self._score_angle(query, a)
            if s > best_s:
                best_a, best_s = float(a), s
        return best_a % 360.0, best_s

    # -- de-rotation with background fill --------------------------------- #
    def _rotate_fill(
        self, img: np.ndarray, angle: float, background: "str | float"
    ) -> np.ndarray:
        """Rotate by -angle and fill the out-of-bounds corners so they don't
        read as anomalies at the patch stage.

        background:
          "template" -> fill from the averaged normal template (best for AD;
                        the corners then look like the real learned background).
          "edge"     -> extend the border pixels outward (skimage mode='edge').
          "auto"     -> flat fill with the estimated background color.
          <float>    -> flat fill with this constant value (grayscale, 0..1).
        """
        if background == "edge":
            return rotate(
                img, -angle, resize=False, preserve_range=True, mode="edge"
            )

        rotated = rotate(
            img, -angle, resize=False, preserve_range=True, mode="constant", cval=0.0
        )
        # Which pixels actually came from the source (vs. out-of-bounds)?
        valid = (
            rotate(
                np.ones_like(img),
                -angle,
                resize=False,
                order=0,
                preserve_range=True,
                mode="constant",
                cval=0.0,
            )
            > 0.5
        )
        out = rotated.copy()
        if background == "template":
            out[~valid] = self.template[~valid]
        elif background == "auto":
            out[~valid] = self._bg_value
        else:  # numeric constant
            out[~valid] = float(background)
        return out

    # -- public API ------------------------------------------------------- #
    def align(
        self, query: np.ndarray, background: "str | float | None" = None
    ) -> AlignResult:
        q = _to_gray_float(query)
        if q.shape != self.shape:
            raise ValueError(
                f"Query shape {q.shape} != template shape {self.shape}. "
                "Resize/crop to the reference size first."
            )

        base = self._coarse_angle(q)

        # Candidate set: the coarse estimate plus its 180-deg twin (Fourier
        # magnitude ambiguity). For "direct" the twin simply scores worse.
        candidates = [base, (base + 180.0) % 360.0]
        scored = [(a, self._score_angle(q, a)) for a in candidates]
        best_angle, best_score = max(scored, key=lambda t: t[1])

        if self.refine:
            best_angle, best_score = self._refine_angle(q, best_angle)

        bg = self.background if background is None else background
        aligned = self._rotate_fill(q, best_angle, bg)
        return AlignResult(
            image=aligned,
            angle=best_angle,
            score=best_score,
            candidates=scored,
        )


def center_object(
    img: np.ndarray, bg_value: float, tol: float = 0.12
) -> np.ndarray:
    """Shift the foreground object's center of mass to the image center.

    Use this BEFORE rotation alignment when the object's position varies
    from image to image (rotation about the image center only works well
    when the object is already near that center — see RotationAligner docs).

    Do NOT use this if off-center position is itself an anomaly signal in
    your data; it will erase that signal.

    Parameters
    ----------
    img      : grayscale float image in [0, 1].
    bg_value : the background gray level (e.g. aligner._bg_value).
    tol      : pixels farther than this from bg_value count as foreground.
    """
    from scipy.ndimage import center_of_mass
    from scipy.ndimage import shift as nd_shift

    fg = np.abs(img - bg_value) > tol
    if fg.sum() < 10:  # nothing clearly foreground -> leave as-is
        return img
    cy, cx = center_of_mass(fg)
    h, w = img.shape
    return nd_shift(
        img, ((h - 1) / 2.0 - cy, (w - 1) / 2.0 - cx), mode="constant", cval=bg_value
    )


# --------------------------------------------------------------------------- #
# Simple one-call entry point: normal images + one input image -> result image
# --------------------------------------------------------------------------- #
def align_image(
    normal_paths: "list[str] | str",
    input_path: str,
    output_path: str,
    size: tuple[int, int] = (256, 256),
    background: "str | float" = "template",
    center: bool = False,
) -> AlignResult:
    """
    The simplest possible entry point.

    Parameters
    ----------
    normal_paths : list of file paths to your aligned normal images,
                   or a single folder path containing them.
    input_path   : path to the test image you want to de-rotate.
    output_path  : where to save the aligned result (png/jpg by extension).
    size         : (H, W) every image is resized to before processing.
    background   : how to fill the corners exposed by rotation
                   ("template" is the safe default for AD).
    center       : if True, recenter the object (via center_object) in both
                   the normals and the input BEFORE rotation alignment.
                   Turn this on only if object position varies between
                   images and that position is NOT itself an anomaly signal.

    Returns
    -------
    AlignResult (also written to disk at `output_path`).

    Example
    -------
    >>> res = align_image(
    ...     normal_paths=['/content/normal.png'],
    ...     input_path='/content/1.png',
    ...     output_path='/content/1_aligned.png',
    ... )
    >>> print(res.angle, res.score)
    """
    from skimage.io import imsave

    if isinstance(normal_paths, str):
        normals = load_images_from_dir(normal_paths, size=size)
    else:
        normals = [
            sk_resize(imread(p), size, preserve_range=True, anti_aliasing=True)
            for p in normal_paths
        ]

    test = sk_resize(imread(input_path), size, preserve_range=True, anti_aliasing=True)

    if center:
        normals = [_to_gray_float(n) for n in normals]
        rough_bg = float(np.median(normals[0]))  # crude estimate, just for centering
        normals = [center_object(n, rough_bg) for n in normals]
        test = center_object(_to_gray_float(test), rough_bg)

    aligner = RotationAligner(normals, background=background)
    result = aligner.align(test)

    imsave(output_path, (np.clip(result.image, 0, 1) * 255).astype(np.uint8))
    print(f"angle={result.angle:.1f}°  score={result.score:.3f}  -> saved: {output_path}")
    return result


# --------------------------------------------------------------------------- #
# Array in, array out: no files, just images.
#
#     from rotation_align import to_normal_pose as rotate
#     x = rotate(x, normals=[normal1, normal2, ...], center=False)
#
# --------------------------------------------------------------------------- #
_aligner_cache: dict = {}


def to_normal_pose(
    x: np.ndarray,
    normals: "list[np.ndarray]",
    center: bool = False,
    background: "str | float" = "template",
) -> np.ndarray:
    """
    Take an input image `x` and the list of normal reference images, and return
    the de-rotated image as a numpy array (grayscale float, values in [0, 1]) —
    same shape as `normals[0]`. No files are read or written.

    Parameters
    ----------
    x        : the input image to correct (array; any size/mode, gets resized
               to match `normals[0]`'s shape).
    normals  : list of aligned normal reference images (arrays).
    center   : if True, recenter the object (via center_object) in both the
               normals and `x` before rotation alignment. Only turn this on if
               object position varies between images AND that position is not
               itself an anomaly signal (see RotationAligner docs).
    background : how to fill corners exposed by rotation ("template" default).

    Example
    -------
    >>> from rotation_align import to_normal_pose as rotate
    >>> x = rotate(x, normals=[normal_img])            # x is now canonical pose
    """
    shape = _to_gray_float(normals[0]).shape
    normals_g = [
        sk_resize(n, shape, preserve_range=True, anti_aliasing=True) for n in normals
    ]
    x_g = sk_resize(x, shape, preserve_range=True, anti_aliasing=True)

    if center:
        normals_g = [_to_gray_float(n) for n in normals_g]
        rough_bg = float(np.median(normals_g[0]))
        normals_g = [center_object(n, rough_bg) for n in normals_g]
        x_g = center_object(_to_gray_float(x_g), rough_bg)

    # Cache the aligner per (normals identity, center, background) so repeated
    # calls on the same normal set (e.g. looping over many test images) don't
    # rebuild the template/polar transform every time.
    cache_key = (id(normals), center, background, shape)
    aligner = _aligner_cache.get(cache_key)
    if aligner is None:
        aligner = RotationAligner(normals_g, background=background)
        _aligner_cache[cache_key] = aligner

    return aligner.align(x_g).image


# --------------------------------------------------------------------------- #
# Self-test: recover a known rotation on a synthetic asymmetric pattern.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from skimage.data import camera
    from skimage.transform import resize

    img = resize(camera().astype(np.float64) / 255.0, (256, 256))

    # Four "normal" refs: same pose, mild per-sample noise.
    rng = np.random.default_rng(0)
    refs = [np.clip(img + rng.normal(0, 0.01, img.shape), 0, 1) for _ in range(4)]

    aligner = RotationAligner(refs, method="fourier", refine=True)

    print(f"{'true':>7} {'est':>7} {'err':>6} {'score':>6}")
    for true_angle in [0, 12, 37, 90, 143, 200, 271, 330]:
        q = rotate(img, true_angle, resize=False, preserve_range=True)
        res = aligner.align(q)
        # est angle should match true_angle (mod 360); wrap error to [-180,180]
        err = (res.angle - true_angle + 180) % 360 - 180
        print(f"{true_angle:7.1f} {res.angle:7.1f} {err:6.1f} {res.score:6.3f}")
