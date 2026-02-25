import cv2
import numpy as np
from app.utils.omr_engine import _preprocess_image, _find_corner_markers, _warp_perspective
from extract_dynamic_layout import get_dynamic_zones

img_path = "omrsheet.png"
img = cv2.imread(img_path)
blurred = _preprocess_image(img)
markers = _find_corner_markers(blurred)
warped = _warp_perspective(img, markers)

zones = get_dynamic_zones(warped)

# Fill Roll (Draw bubbles for 1, 2, 3, 4, 5, 6)
if zones["roll"]:
    rx, ry, rw, rh = zones["roll"]
    cell_w = rw // 6
    cell_h = rh // 10
    for i, digit in enumerate([1, 2, 3, 4, 5, 6]):
        cx = rx + i * cell_w + cell_w // 2
        cy = ry + digit * cell_h + cell_h // 2
        cv2.circle(warped, (cx, cy), 15, (0, 0, 0), -1)

# Fill Set Code (Draw bubble for "খ" which is index 1 out of 6)
if zones["set_code"]:
    sx, sy, sw, sh = zones["set_code"]
    cell_w = sw // 6
    cell_h = sh
    cx = sx + 1 * cell_w + cell_w // 2
    cy = sy + cell_h // 2
    cv2.circle(warped, (cx, cy), 15, (0, 0, 0), -1)

# Fill Answers (Pattern 0, 1, 2, 3)
cols = zones["columns"]
if cols:
    q_per_col = (100 + len(cols) - 1) // len(cols)
    if len(cols) == 4: q_per_col = 25
    
    h, w = warped.shape[:2]
    
    for q in range(100):
        col_idx = q // q_per_col
        row_idx = q % q_per_col
        
        if col_idx < len(cols):
            cx_col, cy_col, cw, ch = cols[col_idx]
            text_w = int(30 * (w / 670.0))
            opt_list_w = cw - text_w
            opt_w = opt_list_w // 4
            opt_h = ch // (q_per_col + 1)
            
            ans = q % 4
            cx = cx_col + text_w + ans * opt_w + opt_w // 2
            cy = cy_col + (row_idx + 1) * opt_h + opt_h // 2
            
            cv2.circle(warped, (cx, cy), 15, (0, 0, 0), -1)

cv2.imwrite("simulated_marked.jpg", warped)
print("Saved simulated_marked.jpg")
