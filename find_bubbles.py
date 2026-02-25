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

# Find all contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

circles = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(cnt)
    
    # A bubble in 2480x3508 warped image is roughly 40-70px wide/tall
    # Area of a 55x55 circle is pi * 27.5^2 = ~2375
    # A standard bubble in the correctly warped 2480px image is ~75x75
    # Area ~ 5600. So we filter tightly around this.
    if 3000 < area < 8000:
        if 0.8 < aspect_ratio < 1.2:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if 0.8 < circularity <= 1.2:
                circles.append((x, y, w, h))

# Deduplicate circles that are very close to each other (inner/outer contour matching)
unique_circles = []
for x, y, w, h in circles:
    is_dup = False
    for cx, cy, cw, ch in unique_circles:
        if abs(x-cx) < 10 and abs(y-cy) < 10:
            is_dup = True
            break
    if not is_dup:
        unique_circles.append((x, y, w, h))

print(f"Detected {len(unique_circles)} unique perfect bubbles.")
for i, (x, y, w, h) in enumerate(unique_circles[:10]):
    print(f"Bubble {i}: X={x}, Y={y}, W={w}, H={h}")

# Draw them
debug_img = warped.copy()
for x, y, w, h in unique_circles:
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 3)

cv2.imwrite("detected_unique_circles.jpg", debug_img)
print("Saved detected_unique_circles.jpg")
