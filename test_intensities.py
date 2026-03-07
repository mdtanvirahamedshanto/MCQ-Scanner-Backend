import cv2
import numpy as np
from app.utils.omr_engine import extract_omr_roi

def check_intensities():
    img = cv2.imread("mcq.png")
    warped_color = extract_omr_roi(img)
    if warped_color is None:
        print("Could not warp")
        return
        
    gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    start_x_left = 680
    start_y = 665
    row_height = 100
    bubble_w, bubble_h = 75, 75
    gap_x = 120
    start_x_right = 1438

    for q in range(20):
        if q < 10:
            y = start_y + q * row_height
            x_start = start_x_left
        else:
            y = start_y + (q - 10) * row_height
            x_start = start_x_right

        scores = []
        for opt in range(4):
            x = x_start + opt * gap_x
            roi = binary[y:y+bubble_h, x:x+bubble_w]
            score = cv2.countNonZero(roi)
            scores.append(score)
        
        print(f"Q{q+1}: scores={scores}")

check_intensities()
