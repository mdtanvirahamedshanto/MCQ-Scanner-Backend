#!/usr/bin/env python3
"""Train/calibrate the 20Q OMR profile from sample sheet images."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from app.utils.omr_engine import (
    _find_corner_markers,
    _get_bubble_density,
    _preprocess_image,
    _warp_perspective,
)

DEFAULT_REF_W = 2480
DEFAULT_REF_H = 2289
DEFAULT_Y_NORM = [
    925 / DEFAULT_REF_H,
    1032 / DEFAULT_REF_H,
    1143 / DEFAULT_REF_H,
    1252 / DEFAULT_REF_H,
    1360 / DEFAULT_REF_H,
    1465 / DEFAULT_REF_H,
    1580 / DEFAULT_REF_H,
    1688 / DEFAULT_REF_H,
    1792 / DEFAULT_REF_H,
    1904 / DEFAULT_REF_H,
]
DEFAULT_X_NORM = [
    [393 / DEFAULT_REF_W, 627 / DEFAULT_REF_W, 856 / DEFAULT_REF_W, 1094 / DEFAULT_REF_W],
    [1067 / DEFAULT_REF_W, 1301 / DEFAULT_REF_W, 1530 / DEFAULT_REF_W, 1768 / DEFAULT_REF_W],
    [1741 / DEFAULT_REF_W, 1975 / DEFAULT_REF_W, 2209 / DEFAULT_REF_W, 2443 / DEFAULT_REF_W],
]


def _snap_to_bubble_center(
    binary: np.ndarray,
    expected_x: int,
    expected_y: int,
    search_radius: int,
) -> Tuple[int, int]:
    """Find closest bubble-like contour center near expected point."""
    h, w = binary.shape
    x1 = max(0, expected_x - search_radius)
    y1 = max(0, expected_y - search_radius)
    x2 = min(w, expected_x + search_radius)
    y2 = min(h, expected_y + search_radius)

    if x2 <= x1 or y2 <= y1:
        return expected_x, expected_y

    roi = binary[y1:y2, x1:x2]
    contours, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    local_x = expected_x - x1
    local_y = expected_y - y1
    best = None
    best_score = float("inf")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40 or area > 6000:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 2.5:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4 * np.pi * (area / (peri * peri))
        if circularity < 0.08:
            continue

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        dist = float(np.hypot(cx - local_x, cy - local_y))
        size_penalty = abs(((bw + bh) / 2.0) - 36.0) * 0.4
        score = dist + size_penalty

        if score < best_score:
            best_score = score
            best = (int(round(cx)), int(round(cy)))

    if best is None:
        return expected_x, expected_y

    return x1 + best[0], y1 + best[1]


def _estimate_mark_threshold(densities: List[float]) -> float:
    """Estimate mark threshold from density distribution using 1D k-means."""
    if len(densities) < 30:
        return 0.17

    arr = np.array(densities, dtype=np.float32)
    c1 = float(np.percentile(arr, 30))
    c2 = float(np.percentile(arr, 90))

    if abs(c2 - c1) < 1e-3:
        return 0.17

    for _ in range(25):
        d1 = np.abs(arr - c1)
        d2 = np.abs(arr - c2)
        g1 = arr[d1 <= d2]
        g2 = arr[d1 > d2]
        if len(g1) == 0 or len(g2) == 0:
            break
        n1 = float(np.mean(g1))
        n2 = float(np.mean(g2))
        if abs(n1 - c1) + abs(n2 - c2) < 1e-4:
            c1, c2 = n1, n2
            break
        c1, c2 = n1, n2

    lo, hi = sorted((c1, c2))
    if hi - lo < 0.04:
        return 0.17

    return float(np.clip((lo + hi) * 0.5, 0.12, 0.30))


def _calibrate_single_image(path: Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not load image: {path}")

    blurred = _preprocess_image(img)
    markers = _find_corner_markers(blurred)
    if markers is None:
        raise RuntimeError(f"Could not detect corner markers: {path}")

    warped = _warp_perspective(img, markers, target_width=DEFAULT_REF_W)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    h, w = gray.shape
    search_radius = max(35, int(round(min(h, w) * 0.03)))
    roi_half_w = max(12, int(round(w * (20 / DEFAULT_REF_W))))
    roi_half_h = max(12, int(round(h * (20 / DEFAULT_REF_H))))

    row_y: List[List[float]] = [[] for _ in range(10)]
    col_x: List[List[List[float]]] = [[[] for _ in range(4)] for _ in range(3)]
    densities: List[float] = []

    for row in range(10):
        expected_y = int(round(DEFAULT_Y_NORM[row] * h))
        for col in range(3):
            for opt in range(4):
                expected_x = int(round(DEFAULT_X_NORM[col][opt] * w))
                cx, cy = _snap_to_bubble_center(binary, expected_x, expected_y, search_radius)

                row_y[row].append(cy / float(h))
                col_x[col][opt].append(cx / float(w))

                y1 = max(0, cy - roi_half_h)
                y2 = min(h, cy + roi_half_h)
                x1 = max(0, cx - roi_half_w)
                x2 = min(w, cx + roi_half_w)
                roi = gray[y1:y2, x1:x2]
                densities.append(_get_bubble_density(roi))

    y_centers_norm = [float(np.median(vals)) if vals else DEFAULT_Y_NORM[i] for i, vals in enumerate(row_y)]
    x_cols_norm = []
    for col in range(3):
        col_vals = []
        for opt in range(4):
            vals = col_x[col][opt]
            col_vals.append(float(np.median(vals)) if vals else DEFAULT_X_NORM[col][opt])
        x_cols_norm.append(col_vals)

    return {
        "image": path.name,
        "warped_width": int(w),
        "warped_height": int(h),
        "y_centers_norm": y_centers_norm,
        "x_cols_norm": x_cols_norm,
        "densities": densities,
    }


def train_profile(images: List[Path]) -> dict:
    samples = []
    for path in images:
        sample = _calibrate_single_image(path)
        samples.append(sample)
        print(f"Calibrated: {path.name} -> warped {sample['warped_width']}x{sample['warped_height']}")

    final_y = []
    for row in range(10):
        vals = [s["y_centers_norm"][row] for s in samples]
        median_val = float(np.median(vals))
        # Keep learned rows close to known template geometry.
        blended = (DEFAULT_Y_NORM[row] * 0.7) + (median_val * 0.3)
        final_y.append(float(np.clip(blended, DEFAULT_Y_NORM[row] - 0.02, DEFAULT_Y_NORM[row] + 0.02)))

    # Enforce strictly increasing rows and healthy row gaps.
    for i in range(1, len(final_y)):
        min_allowed = final_y[i - 1] + 0.025
        if final_y[i] < min_allowed:
            final_y[i] = min_allowed

    final_x = []
    for col in range(3):
        col_vals = []
        for opt in range(4):
            vals = [s["x_cols_norm"][col][opt] for s in samples]
            median_val = float(np.median(vals))
            blended = (DEFAULT_X_NORM[col][opt] * 0.75) + (median_val * 0.25)
            col_vals.append(float(np.clip(blended, DEFAULT_X_NORM[col][opt] - 0.02, DEFAULT_X_NORM[col][opt] + 0.02)))
        final_x.append(col_vals)

    all_densities: List[float] = []
    for s in samples:
        all_densities.extend(s["densities"])

    threshold = _estimate_mark_threshold(all_densities)
    # Keep threshold conservative for mobile/camscanner photos.
    threshold = float(np.clip(threshold, 0.15, 0.17))

    profile = {
        "version": 1,
        "template_type": "20q_mcq_png",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_images": [s["image"] for s in samples],
        "y_centers_norm": final_y,
        "x_cols_norm": final_x,
        "roi_half_w_norm": 20 / DEFAULT_REF_W,
        "roi_half_h_norm": 20 / DEFAULT_REF_H,
        "mark_threshold": threshold,
        "ambiguity_second_min": 0.18,
        "ambiguity_gap_max": 0.06,
        "ambiguity_peak_cap": 0.80,
        "training_stats": {
            "total_samples": len(samples),
            "density_min": float(np.min(all_densities)) if all_densities else 0.0,
            "density_p50": float(np.percentile(all_densities, 50)) if all_densities else 0.0,
            "density_p90": float(np.percentile(all_densities, 90)) if all_densities else 0.0,
            "density_max": float(np.max(all_densities)) if all_densities else 0.0,
        },
    }
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 20Q OMR profile from sample images")
    parser.add_argument(
        "--images",
        nargs="+",
        default=["rawmcq.jpeg", "camscannermcq.jpeg", "mcq.png"],
        help="Training image paths (default: rawmcq.jpeg camscannermcq.jpeg mcq.png)",
    )
    parser.add_argument(
        "--output",
        default="app/utils/profiles/20q_mcq_png_profile.json",
        help="Output profile path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = [Path(p) for p in args.images]

    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        print("Missing images:")
        for p in missing:
            print(f"  - {p}")
        return 1

    try:
        profile = train_profile(image_paths)
    except Exception as exc:
        print(f"Training failed: {exc}")
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Saved profile: {output}")
    print(f"Mark threshold: {profile['mark_threshold']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
