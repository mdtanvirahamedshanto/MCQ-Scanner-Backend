import cv2
import numpy as np

def _get_bubble_density(roi: np.ndarray) -> float:
    if roi.size == 0:
        return 0.0
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return np.sum(binary > 0) / binary.size

def test_density(img_path):
    img = cv2.imread(img_path)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    target_width = 2480
    aspect = gray.shape[0] / float(gray.shape[1])
    target_h = int(target_width * aspect)
    warped = cv2.resize(gray, (target_width, target_h))

    from app.utils.omr_engine import _get_dynamic_zones
    zones = _get_dynamic_zones(warped)
    
    raw_columns = zones["columns"]
    columns = []
    
    for col in raw_columns:
        cx, cy, cw, ch = col
        aspect_ratio = cw / float(ch + 1e-6)
        if aspect_ratio > 0.8:
            num_merged = 3
            step_w = cw // num_merged
            for i in range(num_merged):
                columns.append((cx + i * step_w, cy, step_w, ch))
        else:
            columns.append(col)
            
    questions_per_column_local = 10
    total_questions = 10
    
    for q in range(total_questions):
        col_idx = q // questions_per_column_local
        row_idx = q % questions_per_column_local
        
        if col_idx < len(columns):
            col_x, col_y, col_w, col_h = columns[col_idx]
            
            text_col_w = int(30 * (warped.shape[1] / 670.0))
            opt_list_w = col_w - text_col_w
            opt_w_adj = opt_list_w // 4
            
            opt_h = col_h // (questions_per_column_local + 1)
            
            bubble_grid_x = col_x + text_col_w
            y_start = col_y + (row_idx + 1) * opt_h

            densities = []
            print(f"Q{q+1}:")
            for opt in range(4):
                x = bubble_grid_x + opt * opt_w_adj
                cx_opt, cy_opt = x + opt_w_adj // 2, y_start + opt_h // 2
                
                # In omr_engine.py, it uses r=20 for extraction
                r_x, r_y = 25, 20
                roi = warped[max(0, cy_opt - r_y):min(target_h, cy_opt + r_y), max(0, cx_opt - r_x):min(target_width, cx_opt + r_x)]
                d = _get_bubble_density(roi)
                densities.append(d)
                print(f"  Opt {opt} density: {d:.3f}")
                
            sorted_d = sorted(densities, reverse=True)
            if sorted_d[0] < 0.20:
                print("  => WOULD BE: -1 (Unmarked, < 0.20)")
            elif len(sorted_d) > 1 and sorted_d[1] > 0.20 and (sorted_d[0] - sorted_d[1] < 0.10):
                print("  => WOULD BE: -1 (Ambiguous)")
            else:
                print(f"  => WOULD SELECT: {densities.index(sorted_d[0])}")

test_density('mcq.jpeg')
