"""
OptiMark OMR Engine - Robust OMR processing using OpenCV, NumPy, Imutils.

Implements:
- Four corner marker detection (black squares) for Perspective Transform
- Preprocessing: Grayscale, Gaussian Blur, Canny Edge, Adaptive Thresholding
- Zone segmentation: Roll Number (top-left), Set Code, MCQ Grid (60 questions)
- Pixel Counting / Contour method: bubble marked if highest black pixel density in row
- Supports Bengali set codes: ক, খ, গ, ঘ
- Error handling: "No bubbles detected", "Image too blurry"
"""

import cv2
import numpy as np
from imutils.perspective import four_point_transform
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass, field


# --- Error codes for robust handling ---
class OMRProcessingError(Exception):
    """Base exception for OMR processing errors."""

    pass


class NoBubblesDetectedError(OMRProcessingError):
    """Raised when no bubbles could be detected in the image."""

    pass


class ImageTooBlurryError(OMRProcessingError):
    """Raised when image is too blurry for reliable processing."""

    pass


@dataclass
class OMRResult:
    """Structured result from OMR processing."""

    roll_number: str = ""
    set_code: str = ""
    answers: List[int] = field(default_factory=list)  # 0-3 for each question
    success: bool = False
    error_message: str = ""


# --- OMR Sheet Layout Constants ---
SHEET_WIDTH = 2480
SHEET_HEIGHT = 3508
MARKER_MIN_AREA_RATIO = 0.0005
MARKER_MAX_AREA_RATIO = 0.015
QUESTIONS_PER_COLUMN = 15
OPTIONS_PER_QUESTION = 4
TOTAL_QUESTIONS = 60

# Bengali set codes (ক=0, খ=1, গ=2, ঘ=3, ঙ=4, চ=5) - maps index to character
BENGALI_SET_CODES = ["ক", "খ", "গ", "ঘ", "ঙ", "চ"]
ENGLISH_SET_CODES = ["A", "B", "C", "D", "E", "F"]

# Blur detection threshold (Laplacian variance) - lower = blurrier
BLUR_THRESHOLD = 20

def _get_dynamic_zones(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 80000 < w * h < 2500000:
            boxes.append((x, y, w, h))
            
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    print(f"DEBUG _get_dynamic_zones: {len(boxes)} initial boxes found (area 80k-2.5m)")
    for idx, b in enumerate(boxes):
        print(f"  Raw Box {idx}: {b}")
    filtered_boxes = []
    for b in boxes:
        x, y, w, h = b
        if w > 2000 or h < 200: 
            continue
            
        cx, cy = x + w/2, y + h/2
        is_inside = False
        for fb in filtered_boxes:
            fx, fy, fw, fh = fb
            if fx - 100 < cx < fx+fw + 100 and fy - 100 < cy < fy+fh + 100:
                is_inside = True
                break
        if not is_inside:
            filtered_boxes.append(b)
            
    top_tables = [b for b in filtered_boxes if b[1] < 1400 and b[3] > 300]
    bottom_tables = [b for b in filtered_boxes if b[1] >= 1400 and b[3] > 300]
    print(f"DEBUG _get_dynamic_zones: found {len(top_tables)} top tables, {len(bottom_tables)} bottom tables")
    for idx, b in enumerate(bottom_tables):
        print(f"  Bottom table {idx}: {b}")
    
    top_tables.sort(key=lambda b: b[0])
    
    roll_box = None
    set_code_box = None
    
    if len(top_tables) >= 3:
        set_code_box = top_tables[-1] # Right-most
        roll_box = max(top_tables, key=lambda b: b[2]) # Widest
        
    bottom_tables.sort(key=lambda b: b[0])
    
    return {
        "roll": roll_box,
        "set_code": set_code_box,
        "columns": bottom_tables
    }

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _check_image_blur(gray: np.ndarray) -> bool:
    """
    Check if image is too blurry using Laplacian variance.
    Returns True if image is acceptable, False if too blurry.
    """
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var >= BLUR_THRESHOLD


def _preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Grayscale
    2. Gaussian Blur (noise reduction)
    3. Adaptive Thresholding
    Also used for edge detection: Canny after Gaussian Blur.
    """
    # Step 1: Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Step 2: Gaussian Blur - reduces noise for better edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred


def _find_corner_markers(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the four black square corner markers using:
    - Canny Edge Detection
    - Contour detection
    Returns ordered points: top-left, top-right, bottom-right, bottom-left.
    """
    # Adaptive Threshold - robust for varying lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150)

    # Combine: use both for robustness
    combined = cv2.bitwise_or(thresh, edges)
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape
    total_area = h * w
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MARKER_MIN_AREA_RATIO * total_area < area < MARKER_MAX_AREA_RATIO * total_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w_rect, h_rect = cv2.boundingRect(approx)
                aspect = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1e-6)
                if 0.5 < aspect < 2.0:
                    candidates.append((area, approx))

    candidates.sort(key=lambda x: x[0], reverse=True)
    centers = []
    for c in candidates:
        approx = c[1] # shape (4, 1, 2)
        pts_2d = approx.reshape(-1, 2)
        center = pts_2d.mean(axis=0)
        
        # Filter duplicates (e.g., inner/outer contours of the same square)
        is_duplicate = False
        for existing in centers:
            if np.linalg.norm(center - existing) < 50:
                is_duplicate = True
                break
                
        if not is_duplicate:
            centers.append(center)
            if len(centers) == 10:  # Take up to top 10 distinct shapes
                break

    if len(centers) < 4:
        return None
        
    import itertools
    max_quad_area = 0
    best_quad = None
    
    for quad_indices in itertools.combinations(range(len(centers)), 4):
        quad_pts = np.array([centers[i] for i in quad_indices], dtype=np.float32)
        ordered = _order_points(quad_pts)
        
        # Calculate polygon area of the 4 points
        # area = 0.5 * |(x1y2 - y1x2) + (x2y3 - y2x3) + (x3y4 - y3x4) + (x4y1 - y4x1)|
        x, y = ordered[:, 0], ordered[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        if area > max_quad_area:
            max_quad_area = area
            best_quad = ordered
            
    if best_quad is None:
        return None
        
    # The markers must enclose a significant portion of the document (at least 45%).
    # If it's less, it's a false positive (e.g. internal checkbox grid).
    if max_quad_area < 0.45 * total_area:
        return None

    return best_quad.astype(np.float32)

def _warp_perspective(img: np.ndarray, src_pts: np.ndarray, target_width: int = SHEET_WIDTH) -> np.ndarray:
    """Apply perspective transform for top-down view using imutils."""
    # src_pts are top-left, top-right, bottom-right, bottom-left
    tl, tr, br, bl = src_pts
    
    # Calculate widths and heights
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width_px = max(int(width_top), int(width_bottom))
    
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)
    max_height_px = max(int(height_left), int(height_right))
    
    if max_width_px == 0:
        return img
        
    aspect_ratio = max_height_px / max_width_px
    target_height = int(target_width * aspect_ratio)

    dst_pts = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (target_width, target_height))
    return warped


def _get_bubble_density(roi: np.ndarray) -> float:
    """
    Pixel counting: return ratio of dark pixels in ROI.
    Higher = more filled bubble.
    """
    if roi.size == 0:
        return 0.0
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return np.sum(binary > 0) / binary.size


def _detect_marked_option_by_density(
    img: np.ndarray,
    x_start: int,
    y_start: int,
    num_options: int,
    cell_width: int,
    cell_height: int,
) -> int:
    """
    A bubble is 'marked' if it has the highest density of black pixels
    compared to others in the same row.
    Returns 0 to num_options-1, or -1 if ambiguous/none.
    """
    densities = []
    
    # Debug visualization
    debug_img = None
    if getattr(_detect_marked_option_by_density, "debug_img", None) is None:
        _detect_marked_option_by_density.debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

    for i in range(num_options):
        # We know `cell_width` is around 115px and `cell_height` is around 65px.
        # The bubble itself is roughly 50x50 px in the warped image.
        # So we take the center of the cell and extract a 50x50 box.
        x = x_start + i * cell_width
        cx, cy = x + cell_width // 2, y_start + cell_height // 2
        
        box_radius_x, box_radius_y = 25, 20  # Height is slightly squashed
        
        roi_x1, roi_x2 = max(0, cx - box_radius_x), min(img.shape[1], cx + box_radius_x)
        roi_y1, roi_y2 = max(0, cy - box_radius_y), min(img.shape[0], cy + box_radius_y)
        
        roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
        densities.append(_get_bubble_density(roi))

    max_density = max(densities) if densities else 0
    if max_density < 0.15:  # Relaxed slightly for lighter pencil marks
        return -1
        
    sorted_d = sorted(densities, reverse=True)
    
    # If the second highest is very close to the highest and is also fairly dark, it's a double mark (ambiguous)
    if len(sorted_d) > 1 and sorted_d[1] > 0.20 and (sorted_d[0] - sorted_d[1] < 0.10):
        return -1  # Ambiguous double-mark
        
    return densities.index(sorted_d[0])


def _read_roll_number(
    img: np.ndarray,
    x_start: int,
    y_start: int,
    region_width: int,
    region_height: int,
    num_digits: int = 6,
) -> str:
    """
    Read roll number from top-left zone.
    Layout: num_digits columns, each column has 10 rows (0-9).
    """
    cell_w = region_width // num_digits
    cell_h = region_height // 10
    if cell_w < 2 or cell_h < 2:
        return "?" * num_digits

    result = ""
    for digit_pos in range(num_digits):
        x_base = x_start + digit_pos * cell_w
        densities = []
        for opt in range(10):
            y = y_start + opt * cell_h
            roi = img[y + 1 : y + cell_h - 1, x_base + 1 : x_base + cell_w - 1]
            densities.append(_get_bubble_density(roi))
        max_d = max(densities)
        if max_d >= 0.15: # Lowered threshold to match MCQ answers
            sorted_d = sorted(densities, reverse=True)
            if len(sorted_d) > 1 and sorted_d[1] > 0.15 and (sorted_d[0] - sorted_d[1] < 0.10):
                result += "?" # Ambiguous
            else:
                result += str(densities.index(sorted_d[0]))
        else:
            result += "?"
    return result

def _extract_roll_fallback(warped: np.ndarray, num_digits: int = 6) -> str:
    """
    If the template has corner markers (so it is perspective warped accurately), 
    but table detection fails to find the Roll Number box, we can use the 
    known static location of the Roll grid.
    
    Layout for omrsheet.png:
    - Y = 14% to 37%
    - X = 64% to 92%
    """
def _extract_roll_fallback(warped: np.ndarray, num_digits: int = 6) -> str:
    """
    If the template has corner markers (so it is perspective warped accurately), 
    but table detection fails to find the Roll Number box, we can use the 
    known static location of the Roll grid.
    
    Layout for omrsheet.png:
    - Y = 14% to 37%
    - X = 32% to 62%
    """
    h, w = warped.shape[:2]
    top_y = int(h * 0.14)
    bottom_y = int(h * 0.38)
    left_x = int(w * 0.345)
    right_x = int(w * 0.62)
    
    cell_w = (right_x - left_x) // num_digits
    cell_h = (bottom_y - top_y) // 10
    
    result = ""
    for digit_pos in range(num_digits):
        # Center horizontally on each digit column
        x_base = left_x + digit_pos * cell_w
        cx = x_base + cell_w // 2
        
        densities = []
        for opt in range(10):
            y = top_y + opt * cell_h
            cy = y + cell_h // 2
            
            # Very tight round extraction zone
            r_x, r_y = 12, 12
            
            roi_y1, roi_y2 = max(0, cy - r_y), min(h, cy + r_y)
            roi_x1, roi_x2 = max(0, cx - r_x), min(w, cx + r_x)
            
            roi = warped[roi_y1:roi_y2, roi_x1:roi_x2]
            d = _get_bubble_density(roi)
            densities.append(d)
            
        max_d = max(densities)
        if max_d >= 0.11: # Further slightly lowered threshold for light marks
            sorted_d = sorted(densities, reverse=True)
            if len(sorted_d) > 1 and sorted_d[1] > 0.10 and (sorted_d[0] - sorted_d[1] < 0.05):
                result += "?" # Ambiguous
            else:
                result += str(densities.index(sorted_d[0]))
        else:
            result += "?"
            
    return result

def _extract_setcode_fallback(warped: np.ndarray, use_bengali: bool = True) -> str:
    """
    Fallback for Set Code extraction using known static location on the new template.
    Layout: Y = 14% to 37%, X = 66% to 71%
    """
    h, w = warped.shape[:2]
    top_y = int(h * 0.14)
    bottom_y = int(h * 0.38)
    left_x = int(w * 0.66)
    right_x = int(w * 0.71)
    
    return _read_set_code(warped, left_x, top_y, right_x - left_x, bottom_y - top_y, use_bengali)

def _read_set_code(
    img: np.ndarray,
    x_start: int,
    y_start: int,
    region_width: int,
    region_height: int,
    use_bengali: bool = True,
) -> str:
    """
    Read set code from zone (ক, খ, গ, ঘ or A, B, C, D).
    """
    codes = BENGALI_SET_CODES if use_bengali else ENGLISH_SET_CODES
    num_options = len(codes)
    cell_w = region_width // num_options
    cell_h = region_height
    idx = _detect_marked_option_by_density(
        img, x_start, y_start, num_options, cell_w, cell_h
    )
    return codes[idx] if idx >= 0 else "?"

def _extract_normal_omr_answers(warped: np.ndarray, total_questions: int, columns: list = None) -> list:
    answers = [-1] * total_questions
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped.copy()
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    h_img, w_img = warped.shape[:2]

    # Find ALL bubbles
    b_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # PASS 1: Strict constraints for Y-axis (Row Centers)
    # This guarantees we only lock onto unambiguous dark bubbles, ignoring smudges and light artifacts.
    bubble_ys = []
    for cnt in b_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 800 < area < 12000 and 0.5 < (w / (h + 1e-6)) < 2.0:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circ = 4 * np.pi * (cv2.contourArea(cnt) / (peri * peri))
                if 0.4 < circ <= 1.5:
                    bubble_ys.append(y + h // 2)

    # PASS 2: Loose constraints for X-axis (Column offsets)
    # This ensures extremely faint bubbles (like a lightly drawn Option D) are not skipped,
    # preventing the [-4:] array slice from misaligning.
    bubble_xs = []
    for cnt in b_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 200 < area < 15000 and 0.3 < (w / (h + 1e-6)) < 3.0:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circ = 4 * np.pi * (cv2.contourArea(cnt) / (peri * peri))
                if 0.05 < circ <= 2.5:
                    bubble_xs.append(x + w // 2)
                    
    # Cluster Y coordinates (rows)
    bubble_ys.sort()
    y_clusters = []
    if bubble_ys:
        current_cluster = [bubble_ys[0]]
        for i in range(1, len(bubble_ys)):
            if bubble_ys[i] - bubble_ys[i-1] < 25: # 25px radius
                current_cluster.append(bubble_ys[i])
            else:
                y_clusters.append(int(np.median(current_cluster)))
                current_cluster = [bubble_ys[i]]
        y_clusters.append(int(np.median(current_cluster)))
        
    # We only care about Y clusters that have enough bubbles in them
    b_ys_arr = np.array(bubble_ys)
    valid_y_clusters = []
    for cluster_y in y_clusters:
        count = np.sum(np.abs(b_ys_arr - cluster_y) < 25)
        if count >= 3: # At least a few bubbles in this row to be confident it's a row
            valid_y_clusters.append(cluster_y)
            
    row_centers = valid_y_clusters
    print(f"DEBUG fallback: extracted {len(row_centers)} row centers from bubbles.")

    if len(row_centers) == 0:
        return answers
        
    num_cols_estimate = 3 if total_questions % 3 == 0 else (4 if total_questions % 4 == 0 else 3)
    target_rows = total_questions // num_cols_estimate
    
    if len(row_centers) > target_rows and len(row_centers) >= 3:
        # The physical grid will have a highly consistent median gap between consecutive rows.
        # Find the sequence of `target_rows` that are packed together with the most consistent gap.
        diffs = np.diff(row_centers)
        best_seq = []
        best_variance = float('inf')
        
        for i in range(len(row_centers) - target_rows + 1):
            seq = row_centers[i:i+target_rows]
            seq_diffs = np.diff(seq)
            variance = np.var(seq_diffs)
            if variance < best_variance:
                best_variance = variance
                best_seq = seq
                
        # Only accept the sequence if it is highly uniform (variance < 200, stddev < 14px)
        if best_variance < 250:
            row_centers = best_seq
            print(f"DEBUG fallback: Y-axis gap filtered. Kept {len(row_centers)} best uniform rows.")

    rows_count = len(row_centers)
    num_cols = (total_questions + rows_count - 1) // rows_count

    # Cluster X coordinates
    bubble_xs.sort()
    x_clusters = []
    if bubble_xs:
        current_cluster = [bubble_xs[0]]
        for i in range(1, len(bubble_xs)):
            if bubble_xs[i] - bubble_xs[i-1] < 25: 
                current_cluster.append(bubble_xs[i])
            else:
                x_clusters.append(int(np.median(current_cluster)))
                current_cluster = [bubble_xs[i]]
        x_clusters.append(int(np.median(current_cluster)))

    b_xs_arr = np.array(bubble_xs)
    valid_x_clusters = []
    for cluster_x in x_clusters:
        count = np.sum(np.abs(b_xs_arr - cluster_x) < 25)
        if count >= max(1, rows_count * 0.15): # Appears in at least 15% of rows
            valid_x_clusters.append(cluster_x)
            
    x_grid = valid_x_clusters
    print(f"DEBUG fallback: extracted {len(x_grid)} X columns from bubbles.")
    
    if not x_grid:
        return answers
        
    expected_cols = num_cols * 4
    refined_x_grid = []
    
    if columns and len(columns) == num_cols:
        text_col_w = int(30 * (warped.shape[1] / 670.0))
        text_col_w = int(30 * (warped.shape[1] / 670.0))
        for col in columns:
            cx, cy, cw, ch = col
            # 1. Skip the Text column
            valid_x_start = cx + text_col_w + 20
            valid_x_end = cx + cw
            
            raw_block = sorted([x for x in x_grid if valid_x_start <= x <= valid_x_end])
            
            # Remove any artifacts that are too close to be distinct option columns (< 60px)
            block_xs = []
            if raw_block:
                block_xs.append(raw_block[0])
                for rx in raw_block[1:]:
                    if rx - block_xs[-1] >= 60:
                        block_xs.append(rx)
            
            if len(block_xs) >= 2:
                # 2. Mathematically generate exactly 4 spaced coordinates bounded by the physical bubbles!
                min_bx = min(block_xs)
                spacing = np.median(np.diff(block_xs)) if len(block_xs) > 2 else (max(block_xs) - min_bx)
                if spacing == 0: spacing = (cw - text_col_w) // 4
                
                # Extrapolate exactly 4 points anchored from the first option A (min_bx)
                for i in range(4):
                    refined_x_grid.append(int(min_bx + i * spacing))
            else:
                # Total fallback if column is completely blank
                opt_w_adj = (cw - text_col_w) // 4
                for i in range(4):
                    refined_x_grid.append(int(cx + cw * 0.35 + i * opt_w_adj))
                    
        x_grid = refined_x_grid
    else:
        # Generic fallback if no HTML columns
        diffs = np.diff(x_grid)
        gap_indices = np.argsort(diffs)[-(num_cols - 1):] if num_cols > 1 else []
        gap_indices = sorted(gap_indices)
        blocks = []
        start_idx = 0
        for g_idx in gap_indices:
            blocks.append(x_grid[start_idx:g_idx + 1])
            start_idx = g_idx + 1
        blocks.append(x_grid[start_idx:])
        
        for block in blocks:
            if len(block) >= 4:
                refined_x_grid.extend(block[-4:])
            else:
                refined_x_grid.extend(block)
                
        x_grid = refined_x_grid
        
    debug_img = warped.copy()
    if len(debug_img.shape) == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

    # Generate answers based on intersections!
    for q in range(total_questions):
        c_idx = q // rows_count
        r_idx = q % rows_count
        
        if r_idx >= len(row_centers): continue
            
        cy = row_centers[r_idx]
        max_density = 0
        marked_opt = -1
        
        for opt in range(4):
            x_idx = c_idx * 4 + opt
            if x_idx >= len(x_grid): continue
            cx = x_grid[x_idx]
            
            box_r = 18  # Tighter radius to focus on pencil fill, avoiding empty bubble borders
            roi_y1, roi_y2 = max(0, cy - box_r), min(h_img, cy + box_r)
            roi_x1, roi_x2 = max(0, cx - box_r), min(w_img, cx + box_r)
            
            cv2.rectangle(debug_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
            
            # Use raw grayscale inverse mean for physical darkness instead of edge threshold
            crop_gray = 255 - gray[roi_y1:roi_y2, roi_x1:roi_x2]
            d = np.mean(crop_gray) / 255.0
                
            if d > 0.33 and d > max_density:  # Empty bubbles are around 0.20-0.30. Filled are 0.40-0.80
                max_density = d
                marked_opt = opt
                    
        answers[q] = marked_opt
        
    cv2.imwrite("debug_rois_fallback.jpg", debug_img)
    return answers
class OMRProcessor:
    """
    OMR processing class - encapsulates all OMR logic.
    """

    def __init__(
        self,
        sheet_width: int = SHEET_WIDTH,
        sheet_height: int = SHEET_HEIGHT,
        total_questions: int = TOTAL_QUESTIONS,
        use_bengali_set_codes: bool = True,
    ):
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.total_questions = total_questions
        self.use_bengali_set_codes = use_bengali_set_codes

    def process(self, image_path: Union[str, Path]) -> OMRResult:
        """
        Main entry: process OMR sheet image.
        Extracts Roll Number, Set Code, and MCQ answers.
        """
        result = OMRResult()

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                result.error_message = f"Could not load image: {image_path}"
                return result

            # Preprocessing
            blurred = _preprocess_image(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

            # Blur check
            if not _check_image_blur(gray):
                result.error_message = "Image too blurry for reliable processing"
                raise ImageTooBlurryError(result.error_message)

            # Find corner markers and warp
            markers = _find_corner_markers(blurred)
            if markers is None:
                print("DEBUG: No corner markers found, using resize fallback")
                aspect = gray.shape[0] / float(gray.shape[1])
                target_h = int(self.sheet_width * aspect)
                warped = cv2.resize(gray, (self.sheet_width, target_h))
            else:
                print("DEBUG: Corner markers found, applying perspective transform")
                warped = _warp_perspective(img, markers)
                if len(warped.shape) == 3:
                    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            h, w = warped.shape
            print(f"DEBUG: warped shape is {h}x{w}")

            # Extract zones dynamically using HTML table tracking
            zones = _get_dynamic_zones(warped)

            # Zone 1: Roll Number
            if zones["roll"] is not None:
                roll_x, roll_y, roll_region_w, roll_region_h = zones["roll"]
                result.roll_number = _read_roll_number(
                    warped, roll_x, roll_y, roll_region_w, roll_region_h, 6
                )
            elif markers is not None:
                # If we have 4 corner markers, the image is aligned. 
                # We can use static proportions to find the Roll!
                result.roll_number = _extract_roll_fallback(warped, 6)
            else:
                result.roll_number = "N/A"

            # Zone 2: Set Code
            if zones["set_code"] is not None:
                set_region_x, set_region_y, set_region_w, set_region_h = zones["set_code"]
                result.set_code = _read_set_code(
                    warped,
                    set_region_x,
                    set_region_y,
                    set_region_w,
                    set_region_h,
                    self.use_bengali_set_codes,
                )
            elif markers is not None:
                result.set_code = _extract_setcode_fallback(warped, self.use_bengali_set_codes)
            else:
                result.set_code = "N/A"

            # Zone 3: MCQ Grid
            raw_columns = zones["columns"]
            columns = []
            
            # Split overly wide columns (e.g., if dynamic zones merged all 3 columns into 1)
            for col in raw_columns:
                cx, cy, cw, ch = col
                aspect = cw / float(ch + 1e-6)
                if aspect > 0.8: # Likely merged
                    num_merged = 3 # Standard templates usually merge all 3 columns
                    step_w = cw // num_merged
                    for i in range(num_merged):
                        columns.append((cx + i * step_w, cy, step_w, ch))
                else:
                    columns.append(col)
                    
            answers = []
            bubbles_detected = 0

            # Map the columns to standard grid questions
            if len(columns) > 0 and markers is not None:
                # Based on total questions, how many rows per col?
                questions_per_column_local = (self.total_questions + len(columns) - 1) // len(columns)
                
                # However, the physical template layout stays the same regardless of self.total_questions!
                if len(columns) == 3:
                     # For 3 cols, standard sheets are either 3x20 = 60, or 3x10 = 30.
                     if self.total_questions <= 30:
                         questions_per_column_local = 10
                     else:
                         questions_per_column_local = 20
                elif self.total_questions == 100:
                    questions_per_column_local = 25
                elif self.total_questions == 60:
                    questions_per_column_local = 20
                elif self.total_questions == 40:
                    questions_per_column_local = 20

                for q in range(self.total_questions):
                    col_idx = q // questions_per_column_local
                    row_idx = q % questions_per_column_local
                    
                    if col_idx < len(columns):
                        col_x, col_y, col_w, col_h = columns[col_idx]
                        
                        # Within the column, option bubbles start after the question number.
                        # Question number cell is ~30 HTML px -> ~111 px warped.
                        # 4 Options remain.
                        text_col_w = int(30 * (warped.shape[1] / 670.0))
                        opt_list_w = col_w - text_col_w
                        opt_w_adj = opt_list_w // OPTIONS_PER_QUESTION
                        
                        # Add 1 to account for the "প্রশ্ন | উত্তর" header row in the table!
                        opt_h = col_h // (questions_per_column_local + 1)
                        
                        bubble_grid_x = col_x + text_col_w
                        # Add 1 to row_idx because 0-th row is the text header
                        y_start = col_y + (row_idx + 1) * opt_h

                        marked = _detect_marked_option_by_density(
                            warped, bubble_grid_x, y_start, OPTIONS_PER_QUESTION, opt_w_adj, opt_h
                        )
                    else:
                        marked = -1
                        
                    if marked >= 0:
                        bubbles_detected += 1
                    answers.append(marked)
            else:
                print("Fallback: Using proportional matrix mapping for NormalOMRLayout...")
                answers = _extract_normal_omr_answers(warped, self.total_questions, columns)
                bubbles_detected = sum(1 for a in answers if a >= 0)
            
            result.answers = answers

            if bubbles_detected == 0:
                result.error_message = "No bubbles detected in the image"
                raise NoBubblesDetectedError(result.error_message)

            result.success = True

        except (NoBubblesDetectedError, ImageTooBlurryError):
            result.success = False
        except Exception as e:
            result.error_message = str(e)
            result.success = False

        return result


def process_omr_image(
    image_path: Union[str, Path],
    num_questions: int = 60,
    use_bengali_set_codes: bool = True,
) -> OMRResult:
    """
    Convenience function - process OMR sheet image.
    """
    processor = OMRProcessor(
        total_questions=num_questions,
        use_bengali_set_codes=use_bengali_set_codes,
    )
    return processor.process(image_path)


def grade_omr_result(
    omr_result: OMRResult,
    answer_key: Dict[str, int],
    marks_per_question: float = 1.0,
    negative_marking: float = 0.0,
) -> Tuple[int, List[int], float]:
    """
    Grade OMR result against answer key.
    answer_key: {"1": 0, "2": 2, "3": 1, ...} - question_no (str) -> correct_option (0-3)

    Returns:
        (marks_obtained, wrong_answers_list, percentage)
    """
    correct = 0
    wrong_list = []

    for q_no, marked in enumerate(omr_result.answers, start=1):
        key = str(q_no)
        if key not in answer_key:
            continue
        correct_opt = answer_key[key]
        if marked == correct_opt:
            correct += 1
        elif marked >= 0:
            wrong_list.append(q_no)

    total = len(answer_key)
    marks_obtained = max(
        0,
        correct * marks_per_question - len(wrong_list) * negative_marking
    )
    percentage = (marks_obtained / (total * marks_per_question) * 100) if total > 0 else 0.0

    return int(marks_obtained), wrong_list, round(percentage, 2)
