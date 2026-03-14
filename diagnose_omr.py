import cv2
import numpy as np
import os
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

    h, w = img.shape[:2]
    print(f"Image Resolution: {w}x{h}")

    # Preprocessing
    blurred = _preprocess_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Marker Detection
    markers = _find_corner_markers(blurred)
    if markers is not None:
        print("Markers Found!")
        for i, pt in enumerate(markers):
            print(f"  Marker {i}: {pt}")
        
        # Draw markers on original image for visualization
        debug_markers = img.copy()
        for pt in markers:
            cv2.circle(debug_markers, (int(pt[0]), int(pt[1])), 20, (0, 0, 255), -1)
        cv2.imwrite(f"debug_markers_{os.path.basename(path)}.jpg", debug_markers)
        
        # Warp
        warped = _warp_perspective(img, markers)
        cv2.imwrite(f"debug_warped_{os.path.basename(path)}.jpg", warped)
        print(f"Warped Shape: {warped.shape}")
        
        # Zones
        zones = _get_dynamic_zones(warped)
        print(f"Zones: {zones}")
    else:
        print("Markers NOT Found!")
        # Try a direct resize to see if zones can be found without warp
        target_w = 2480
        aspect = h / w
        target_h = int(target_w * aspect)
        resized = cv2.resize(gray, (target_w, target_h))
        zones = _get_dynamic_zones(resized)
        print(f"Zones (Resized): {zones}")

    # Run full process
    proc = OMRProcessor(total_questions=60, template_type="auto")
    res = proc.process(path)
    print(f"Result Success: {res.success}")
    print(f"Result Roll: {res.roll_number}")
    print(f"Result Set: {res.set_code}")
    print(f"Result Error: {res.error_message}")
    bubbles = [a for a in res.answers if a >= 0]
    print(f"Bubbles Detected: {len(bubbles)} / {len(res.answers)}")

if __name__ == "__main__":
    analyze_image("mcq.png")
    analyze_image("rawmcq.jpeg")
