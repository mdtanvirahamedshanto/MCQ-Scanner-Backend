import cv2
import numpy as np
from app.utils.omr_engine import extract_omr_roi

def draw():
    # Load original
    img = cv2.imread("mcq.png")
    warped_color = extract_omr_roi(img)
    if warped_color is None:
        print("Could not warp")
        return
    
    H, W = warped_color.shape[:2]
    # Redraw the grid exactly as defined in process_20q_mcq_png
    # But wait, I can just copy the constants from there
    import importlib.util
    spec = importlib.util.spec_from_file_location("omr", "app/utils/omr_engine.py")
    omr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(omr)

    gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    start_x_left = 680
    start_y = 665
    row_height = 100
    bubble_w, bubble_h = 75, 75
    gap_x = 120

    start_x_right = 1438

    for q in range(10):
        # Left
        y = start_y + q * row_height
        for opt in range(4):
            x = start_x_left + opt * gap_x
            cv2.rectangle(warped_color, (x, y), (x+bubble_w, y+bubble_h), (0,0,255), 4)
            # score = cv2.countNonZero(binary[y:y+bubble_h, x:x+bubble_w])
            # cv2.putText(warped_color, str(score), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        # Right
        for opt in range(4):
            x = start_x_right + opt * gap_x
            cv2.rectangle(warped_color, (x, y), (x+bubble_w, y+bubble_h), (255,0,0), 4)

    cv2.imwrite("debug_boxes_20q.jpg", warped_color)
    print("Saved debug_boxes_20q.jpg")

draw()
