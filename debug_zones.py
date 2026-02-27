import cv2
import numpy as np
from app.utils.omr_engine import _preprocess_image, _get_dynamic_zones

img = cv2.imread('uploads/d00b7e1e-ecf0-4438-9c41-2a56c0196b2a.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
aspect = gray.shape[0] / float(gray.shape[1])
warped = cv2.resize(gray, (2480, int(2480 * aspect)))

# We need to replicate _get_dynamic_zones logic but save an image
thresh = cv2.adaptiveThreshold(
    warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 80000 < w * h < 2500000:
        boxes.append((x, y, w, h))
        cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite("debug_zones_raw.jpg", dbg)

boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
filtered_boxes = []
for b in boxes:
    x, y, w, h = b
    if w > 2000 or h < 200: 
        continue
        
    cx, cy = x + w/2, y + h/2
    is_inside = False
    for fb in filtered_boxes:
        fx, fy, fw, fh = fb
        if fx - 100 < cx < fx+fw + 100 and fy - 100 < cy < fy+fh + 100:
            is_inside = True
            break
    if not is_inside:
        filtered_boxes.append(b)

dbg_filtered = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
for x, y, w, h in filtered_boxes:
    cv2.rectangle(dbg_filtered, (x, y), (x+w, y+h), (0, 0, 255), 4)

cv2.imwrite("debug_zones_filtered.jpg", dbg_filtered)

print(f"Total raw boxes (80k - 2.5m area): {len(boxes)}")
print(f"Filtered boxes: {len(filtered_boxes)}")
for b in filtered_boxes:
    print(b)
