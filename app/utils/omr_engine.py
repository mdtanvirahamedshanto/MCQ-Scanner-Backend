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
BLUR_THRESHOLD = 100

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
            
    top_tables = [b for b in filtered_boxes if b[1] < 1400 and b[3] > 600]
    bottom_tables = [b for b in filtered_boxes if b[1] >= 1400 and b[3] > 600]
    
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
        
        cv2.rectangle(_detect_marked_option_by_density.debug_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,0,255), 2)
        
        roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
        densities.append(_get_bubble_density(roi))

    max_density = max(densities) if densities else 0
    if max_density < 0.25:  # No bubble sufficiently filled
        return -1
    max_indices = [i for i, d in enumerate(densities) if d == max_density]
    if len(max_indices) != 1:
        return -1  # Ambiguous
    return max_indices[0]


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
        if max_d >= 0.25:
            idx = [i for i, d in enumerate(densities) if d == max_d]
            result += str(idx[0]) if len(idx) == 1 else "?"
        else:
            result += "?"
    return result


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

def _extract_normal_omr_answers(warped: np.ndarray, total_questions: int) -> list:
    answers = [-1] * total_questions
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped.copy()
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    h_img, w_img = warped.shape[:2]

    # 1. Find the Y coordinates using left timing marks
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    t_marks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h + 1e-6)
        if 800 < w * h < 12000 and x < w_img * 0.15: # Target left edge
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(cv2.contourArea(cnt)) / hull_area
                if solidity > 0.9 and 1.0 < aspect_ratio < 2.0:
                    # Also need to make sure the inside is completely dark
                    roi = warped[y:y+h, x:x+w]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                    if np.mean(gray_roi) < 100:
                        t_marks.append((x, y, w, h))

    t_marks.sort(key=lambda m: m[1])
    # Each row is vertically centered to the tracking mark
    raw_row_centers = [y + h // 2 for (x, y, w, h) in t_marks]
    
    # Cluster row centers (radius 20px) to merge duplicate contours of the same mark
    row_centers = []
    if raw_row_centers:
        raw_row_centers.sort()
        current_rc = [raw_row_centers[0]]
        for i in range(1, len(raw_row_centers)):
            if raw_row_centers[i] - raw_row_centers[i-1] < 20:
                current_rc.append(raw_row_centers[i])
            else:
                row_centers.append(int(np.median(current_rc)))
                current_rc = [raw_row_centers[i]]
        row_centers.append(int(np.median(current_rc)))

    print(f"Row Centers ({len(row_centers)}): {row_centers}")
    if len(row_centers) == 0:
        return answers # Failed completely

    rows_count = len(row_centers)
    num_cols = (total_questions + rows_count - 1) // rows_count

    # 2. Find the X coordinates using perfectly printed bubbles
    b_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bubble_xs = []
    for cnt in b_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 3000 < area < 8000 and 0.8 < (w / (h + 1e-6)) < 1.2:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circ = 4 * np.pi * (cv2.contourArea(cnt) / (peri * peri))
                if 0.8 < circ <= 1.2:
                    bubble_xs.append(x + w // 2)

    if len(bubble_xs) < num_cols * 4: # Not enough valid shapes
        return answers

    # Cluster X coordinates (radius 20px)
    bubble_xs.sort()
    clusters = []
    current_cluster = [bubble_xs[0]]
    for i in range(1, len(bubble_xs)):
        if bubble_xs[i] - bubble_xs[i-1] < 20: 
            current_cluster.append(bubble_xs[i])
        else:
            clusters.append(int(np.median(current_cluster)))
            current_cluster = [bubble_xs[i]]
    clusters.append(int(np.median(current_cluster)))

    b_xs_arr = np.array(bubble_xs)
    valid_clusters = []
    for cluster_x in clusters:
        # Require cluster to appear in at least 25% of rows
        count = np.sum(np.abs(b_xs_arr - cluster_x) < 20)
        if count >= max(1, rows_count * 0.25):
            valid_clusters.append(cluster_x)
            
    valid_clusters.sort()
    expected_cols = num_cols * 4
    if len(valid_clusters) < expected_cols:
        return answers
        
    x_grid = valid_clusters[:expected_cols]
    print(f"Total valid clusters: {len(valid_clusters)}. First 16: {valid_clusters[:16]}")
    print(f"Selected x_grid ({len(x_grid)}): {x_grid}")

    # Draw debug points
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
            
            box_r = 25
            roi_y1, roi_y2 = max(0, cy - box_r), min(h_img, cy + box_r)
            roi_x1, roi_x2 = max(0, cx - box_r), min(w_img, cx + box_r)
            
            cv2.rectangle(debug_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
            
            d = _get_bubble_density(warped[roi_y1:roi_y2, roi_x1:roi_x2])
            if d > 0.2:
                print(f"Q{q} Opt{opt}: density={d:.2f}, cx={cx}, cy={cy}")

            if d > 0.25 and d > max_density:
                max_density = d
                marked_opt = opt
                
        answers[q] = marked_opt
        
    cv2.imwrite("debug_rois_normal.jpg", debug_img)
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
                aspect = gray.shape[0] / float(gray.shape[1])
                target_h = int(self.sheet_width * aspect)
                warped = cv2.resize(gray, (self.sheet_width, target_h))
            else:
                warped = _warp_perspective(img, markers)
                if len(warped.shape) == 3:
                    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            h, w = warped.shape

            # Extract zones dynamically using HTML table tracking
            zones = _get_dynamic_zones(warped)

            # Zone 1: Roll Number
            if zones["roll"] is not None:
                roll_x, roll_y, roll_region_w, roll_region_h = zones["roll"]
                result.roll_number = _read_roll_number(
                    warped, roll_x, roll_y, roll_region_w, roll_region_h, 6
                )
            else:
                result.roll_number = "unknown"

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
            else:
                result.set_code = "?"

            # Zone 3: MCQ Grid
            columns = zones["columns"]
            answers = []
            bubbles_detected = 0

            # Map the columns to standard grid questions
            if len(columns) > 0:
                # Based on total questions, how many rows per col?
                questions_per_column_local = (self.total_questions + len(columns) - 1) // len(columns)
                
                # However, for typical 100Q it's 25. For 60Q it's 20.
                if self.total_questions == 100:
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
                        text_col_w = int(30 * (w / 670.0))
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
                answers = _extract_normal_omr_answers(warped, self.total_questions)
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
