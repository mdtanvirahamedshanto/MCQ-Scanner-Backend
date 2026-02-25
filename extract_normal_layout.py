import cv2
import numpy as np
from app.utils.omr_engine import _preprocess_image, _find_corner_markers, _warp_perspective

img = cv2.imread("omrsheet.png")
blurred = _preprocess_image(img)
markers = _find_corner_markers(blurred)
warped = _warp_perspective(img, markers)

cv2.imwrite("warped_normal.jpg", warped)

gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)

# Find giant box (e.g. height > 1000, width > 1500)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

biggest_box = None
max_area = 0

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area > 500000:
        print(f"Large Contours: {x}, {y}, {w}, {h} Area: {area}")
        if area > max_area and w > 1000:
            biggest_box = (x, y, w, h)
            max_area = area

if biggest_box:
    print(f"Giant Bounding Box: {biggest_box}")
    x, y, w, h = biggest_box
    cv2.rectangle(warped, (x, y), (x+w, y+h), (0, 0, 255), 5)
else:
    print("Could not find the 5px bounding box.")

cv2.imwrite("normal_bounding_box.jpg", warped)
print("Saved normal_bounding_box.jpg")
