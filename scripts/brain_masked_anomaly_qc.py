#!/usr/bin/env python
"""
Post-hoc QC: restrict autoDDPM anomaly maps to a T1-derived brain mask.

Does not re-run the model. Writes:
  - brain mask NIfTI (optional rim erosion to down-weight skull edge)
  - stats computed only inside mask
  - largest connected-component sizes for (anomaly > t) & mask
  - middle-slice PNG: original / anomaly / masked anomaly
"""

from __future__ import annotations

import argparse
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import filters, morphology


def _largest_cc_size(mask: np.ndarray, structure: np.ndarray) -> int:
    if mask.ndim != 3:
        raise ValueError(f"expected 3D mask, got {mask.shape}")
    if not np.any(mask):
        return 0
    lbl, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return 0
    sizes = ndimage.sum(mask, lbl, index=np.arange(1, n + 1))
    return int(np.max(sizes))


def brain_mask_t1(
    vol: np.ndarray,
    closing_radius: int = 2,
    opening_radius: int = 1,
    rim_erode_iters: int = 0,
) -> np.ndarray:
    """
    Lightweight skull-adjacent mask from T1 (no FSL). For QC only.
    """
    data = np.asarray(vol, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected 3D volume, got {data.shape}")

    positive = data > 0
    if not np.any(positive):
        raise ValueError("volume appears empty")

    # Otsu on foreground voxels avoids huge zero background dominating histogram.
    fg_vals = data[positive]
    thr = float(filters.threshold_otsu(fg_vals))
    mask = data > thr

    mask = ndimage.binary_fill_holes(mask)
    if closing_radius > 0:
        mask = morphology.binary_closing(mask, morphology.ball(closing_radius))
    if opening_radius > 0:
        mask = morphology.binary_opening(mask, morphology.ball(opening_radius))

    lbl, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lbl, index=np.arange(1, n + 1))
    keep_label = int(1 + int(np.argmax(sizes)))
    mask = lbl == keep_label

    if rim_erode_iters > 0:
        struct = np.ones((3, 3, 3), dtype=bool)
        for _ in range(rim_erode_iters):
            mask = ndimage.binary_erosion(mask, structure=struct)

    return mask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t1", required=True, help="T1w NIfTI (same grid as anomaly map)")
    p.add_argument("--anomaly", required=True, help="anomaly_map NIfTI from inference_clean.py")
    p.add_argument("--output-dir", required=True, help="directory for mask, stats, PNG")
    p.add_argument(
        "--prefix",
        default=None,
        help="output basename without extension (default: anomaly map stem)",
    )
    p.add_argument("--closing-radius", type=int, default=2)
    p.add_argument("--opening-radius", type=int, default=1)
    p.add_argument(
        "--rim-erode-iters",
        type=int,
        default=2,
        help="binary erosion iterations on mask to strip skull/CSF rim (0 disables)",
    )
    args = p.parse_args()

    t1_img = nib.load(args.t1)
    amap_img = nib.load(args.anomaly)

    t1 = t1_img.get_fdata()
    amap = amap_img.get_fdata().astype(np.float32)

    if t1.shape != amap.shape:
        print(
            f"Error: shape mismatch T1 {t1.shape} vs anomaly {amap.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not np.allclose(t1_img.affine, amap_img.affine, rtol=0, atol=1e-3):
        print(
            "Warning: affines differ between T1 and anomaly map; "
            "assuming same grid anyway (QC only).",
            file=sys.stderr,
        )

    base = args.prefix
    if not base:
        base = os.path.basename(args.anomaly).replace(".nii.gz", "").replace(".nii", "")
        if base.endswith("_anomaly_map"):
            base = base[: -len("_anomaly_map")]

    os.makedirs(args.output_dir, exist_ok=True)

    mask = brain_mask_t1(
        t1,
        closing_radius=args.closing_radius,
        opening_radius=args.opening_radius,
        rim_erode_iters=args.rim_erode_iters,
    )

    mask_path = os.path.join(args.output_dir, f"{base}_brain_mask_qc.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine=t1_img.affine), mask_path)

    inside = mask.astype(bool)
    amap_in = amap[inside]

    lines = [
        "autoDDPM brain-masked anomaly QC (post-hoc)",
        "=" * 50,
        "",
        f"brain_mask_voxels: {int(inside.sum())}",
        f"brain_mask_fraction_of_fov: {float(inside.mean()):.6f}",
        f"rim_erode_iters: {args.rim_erode_iters}",
        "",
        "anomaly statistics inside brain mask:",
        f"  mean: {float(np.mean(amap_in)):.8f}",
        f"  max: {float(np.max(amap_in)):.8f}",
        f"  p85: {float(np.percentile(amap_in, 85)):.8f}",
        f"  p90: {float(np.percentile(amap_in, 90)):.8f}",
        f"  p95: {float(np.percentile(amap_in, 95)):.8f}",
        "",
        "connected components on (anomaly > t) & mask (26-conn):",
    ]

    struct26 = np.ones((3, 3, 3), dtype=bool)
    for t in (0.10, 0.13, 0.20, 0.30, 0.50):
        m = (amap > t) & inside
        cc = _largest_cc_size(m, struct26)
        lines.append(
            f"  thr {t:.2f}: positives {int(m.sum())} ({100.0 * m.mean():.4f}% of FOV), "
            f"largest_cc {cc} voxels"
        )

    stats_path = os.path.join(args.output_dir, f"{base}_stats_brain_masked.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Middle-slice figure (same index as inference_clean.py)
    mid = amap.shape[2] // 2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(t1[:, :, mid], cmap="gray")
    axes[0].set_title("T1 (middle axial)")
    axes[0].axis("off")

    im1 = axes[1].imshow(amap[:, :, mid], cmap="hot")
    axes[1].set_title(f"Anomaly map (max {np.max(amap[:, :, mid]):.3f})")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    masked_slice = np.where(inside[:, :, mid], amap[:, :, mid], 0.0)
    im2 = axes[2].imshow(masked_slice, cmap="hot")
    axes[2].set_title(
        f"Masked anomaly (max {np.max(masked_slice):.3f}; rim_erode={args.rim_erode_iters})"
    )
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    png_path = os.path.join(args.output_dir, f"{base}_middle_slice_brain_masked.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Wrote {mask_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
