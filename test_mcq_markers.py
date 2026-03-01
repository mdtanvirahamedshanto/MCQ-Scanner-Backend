import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.omr_engine import *

def check_markers(img_path):
    print(f"\n--- Checking Markers for: {img_path} ---")
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found.")
        return
        
    from app.utils.omr_engine import _preprocess_image, _find_corner_markers
    blurred = _preprocess_image(img)
    markers = _find_corner_markers(blurred)
    
    if markers is not None:
        print(f"Found markers at:\n{markers}")
        # Draw the markers on the original image
        debug_img = img.copy()
        for pt in markers:
            cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 20, (0, 0, 255), -1)
            
        cv2.imwrite("mcq_markers_debug.jpg", debug_img)
    else:
        print("No markers found.")

check_markers('mcq.jpeg')
