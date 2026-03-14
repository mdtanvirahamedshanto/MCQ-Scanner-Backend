import cv2
import numpy as np
import os
import sys

# Add path to import app modules
sys.path.append('.')

from app.utils.omr_engine import OMRProcessor, _preprocess_image, _find_corner_markers, _warp_perspective, _get_dynamic_zones

def analyze_image(path):
    print(f"\n--- Analyzing {path} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    img = cv2.imread(path)
    if img is None:
        print(f"Failed to load image: {path}")
        return

    base = os.path.basename(path)
    h, w = img.shape[:2]
    print(f"Image Resolution: {w}x{h}")

    # Preprocessing
    blurred = _preprocess_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Blur Detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Blur (Laplacian Var): {laplacian_var:.2f} (Threshold: 20)")

    # Zone Detection Thresholding Visualization
    zone_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    cv2.imwrite(f"debug_thresh_zones_{base}.jpg", zone_thresh)

    # Marker Detection
    markers = _find_corner_markers(blurred)
    if markers is not None:
        print("Markers Found!")
        # Draw markers
        debug_markers = img.copy()
        for pt in markers:
            cv2.circle(debug_markers, (int(pt[0]), int(pt[1])), 20, (0, 0, 255), -1)
        cv2.imwrite(f"debug_markers_{base}.jpg", debug_markers)
        
        # Warp
        warped = _warp_perspective(img, markers)
        cv2.imwrite(f"debug_warped_{base}.jpg", warped)
    else:
        print("Markers NOT Found!")

    # Run full process
    proc = OMRProcessor(total_questions=60, template_type="auto")
    res = proc.process(path)
    
    # Rename the debug_rois_fallback.jpg if it exists
    if os.path.exists("debug_rois_fallback.jpg"):
        os.rename("debug_rois_fallback.jpg", f"debug_rois_{base}.jpg")

    print(f"Result Success: {res.success}")
    print(f"Result Roll: {res.roll_number}")
    print(f"Result Set: {res.set_code}")
    print(f"Result Error: {res.error_message}")
    
    answered = []
    for i, a in enumerate(res.answers):
        if a >= 0:
            answered.append((i+1, ["A","B","C","D"][a]))
    
    print(f"Total Detected: {len(answered)}")
    # print(f"Answers: {answered}")

if __name__ == "__main__":
    analyze_image("mcq.png")
    analyze_image("rawmcq.jpeg")
    analyze_image("camscannermcq.jpeg")
