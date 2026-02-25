import cv2
import numpy as np
from app.utils.omr_engine import _preprocess_image, _find_corner_markers, _warp_perspective

img = cv2.imread("omrsheet.png")
blurred = _preprocess_image(img)
markers = _find_corner_markers(blurred)
warped = _warp_perspective(img, markers)

gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)

# Timing marks are small dark rectangles.
# In React: width: 24px, height: 20px (Top/Bottom)
# In React: width: 16px, height: 10px (Left/Right)
# Let's find all contours that are small and roughly rectangular
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

t_marks = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    aspect_ratio = w / float(h)
    
    # Warped scale: 2480 width for 190mm = 13px/mm. CSS pixels are ~3.8x warped.
    # 24x20 -> ~90x75
    # 16x10 -> ~60x38
    # Let's target area 1000 to 10000
    if 1000 < area < 12000:
        # Check solidity (rectangularity)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(cv2.contourArea(cnt)) / hull_area
            # Timing marks are solid black rectangles
            if solidity > 0.9: 
                # Aspect ratios: Top/Bottom are ~1.2 (24x20), Left/Right are ~1.6 (16x10)
                if 1.0 < aspect_ratio < 2.0:
                    # Also need to make sure the inside is completely dark
                    roi = warped[y:y+h, x:x+w]
                    # To avoid circular dependency, just check pixel intensity
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                    if np.mean(gray_roi) < 100: # Very dark
                        t_marks.append((x, y, w, h))

print(f"Detected {len(t_marks)} potential timing marks.")

t_marks.sort(key=lambda b: b[1]) # Sort by Y
for i, (x, y, w, h) in enumerate(t_marks):
    print(f"Timing Mark {i}: X={x}, Y={y}, W={w}, H={h}")

debug_img = warped.copy()
for x, y, w, h in t_marks:
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 3)

cv2.imwrite("timing_marks.jpg", debug_img)
print("Saved timing_marks.jpg")
