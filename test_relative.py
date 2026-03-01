import cv2
import numpy as np
from app.utils.omr_engine import process_omr_image, _read_roll_number, _read_set_code, _get_dynamic_zones

def test_relative_bounds(img_path):
    print(f"\n--- Analyzing {img_path} ---")
    img = cv2.imread(img_path)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    target_width = 2480
    aspect = gray.shape[0] / float(gray.shape[1])
    target_h = int(target_width * aspect)
    warped = cv2.resize(gray, (target_width, target_h))

    zones = _get_dynamic_zones(warped)
    columns = zones["columns"]
    
    if len(columns) > 0:
        c0 = columns[0]
        col_x, col_y, col_w, col_h = c0
        print(f"MCQ Grid starts at X={col_x}, Y={col_y}, W={col_w}, H={col_h}")
        
        # Estimate Roll Box
        roll_w = int(col_w * 0.7)
        roll_h = int(col_w * 0.6)
        roll_x = col_x
        roll_y = max(0, col_y - roll_h - int(col_w * 0.05))
        roll = _read_roll_number(warped, roll_x, roll_y, roll_w, roll_h)
        print(f"Guessed Roll: {roll}")
        
        # Estimate Set Code
        set_w = int(col_w * 0.25)
        set_h = int(col_w * 0.3)
        set_x = col_x + col_w - set_w
        set_y = max(0, col_y - set_h - int(col_w * 0.05))
        set_code = _read_set_code(warped, set_x, set_y, set_w, set_h)
        print(f"Guessed Set Code: {set_code}")
    else:
        # For fallback (mcq2, mcq3)
        print("MCQ Grid not cleanly detected. Can't guess from tables.")

for img in ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']:
    test_relative_bounds(img)
