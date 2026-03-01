import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.omr_engine import *

def debug_mcq(img_path):
    print(f"\n--- Debugging High-Quality MCQ Image: {img_path} ---")
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found.")
        return
        
    from app.utils.omr_engine import _preprocess_image, _find_corner_markers, _warp_perspective, SHEET_WIDTH
    blurred = _preprocess_image(img)
    markers = _find_corner_markers(blurred)
    if markers is not None:
        warped = _warp_perspective(img, markers)
    else:
        aspect = img.shape[0] / float(img.shape[1])
        target_h = int(SHEET_WIDTH * aspect)
        warped = cv2.resize(img, (SHEET_WIDTH, target_h))
        
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped.copy()
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    h_img, w_img = warped.shape[:2]

    # Find ALL bubbles
    b_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    debug_img = warped.copy()
    if len(debug_img.shape) == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

    valid_bubbles = 0
    for cnt in b_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 100 < area < 20000:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circ = 4 * np.pi * (cv2.contourArea(cnt) / (peri * peri))
                aspect_ratio = w / (h + 1e-6)
                
                # Check if it was accepted by the current rules
                accepted = False
                if 200 < area < 4000 and 0.5 < aspect_ratio < 2.0 and 0.4 < circ <= 1.5:
                    accepted = True
                    valid_bubbles += 1
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    # Draw rejected bubbles in red if they look somewhat like bubbles
                    if 0.5 < aspect_ratio < 2.0 and circ > 0.3 and 100 < area < 10000:
                        print(f"Rejected: Area={area}, Aspect={aspect_ratio:.2f}, Circ={circ:.2f}")
                        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        
    print(f"Total contours: {len(b_contours)}, Valid Bubbles (Green): {valid_bubbles}")
    cv2.imwrite("mcq_bubbles_debug.jpg", debug_img)
    
debug_mcq('mcq.jpeg')
