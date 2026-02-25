import cv2
import numpy as np

img = cv2.imread("warped_omr.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)

# Detect horizontal and vertical lines specifically
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

# Combine lines to form grids
table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.)
_, table_structure = cv2.threshold(table_structure, 128, 255, cv2.THRESH_BINARY)

# Find contours of tables
contours, hierarchy = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bounding_boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area > 100000: # Filter small artifacts
        bounding_boxes.append((x, y, w, h))

# Sort boxes
bounding_boxes.sort(key=lambda b: (b[1], b[0])) # Sort by Y then X

print(f"Found {len(bounding_boxes)} large tables.")
for x, y, w, h in bounding_boxes:
    print(f"Table Box: X={x}, Y={y}, W={w}, H={h} (Area={w*h})")
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)

cv2.imwrite("tables_detected.jpg", img)
print("Saved tables_detected.jpg")
