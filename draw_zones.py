import cv2
import numpy as np
from app.utils.omr_engine import process_omr_image, _preprocess_image, _find_corner_markers, _warp_perspective, SHEET_WIDTH, SHEET_HEIGHT

img = cv2.imread("test_omr.jpg")
blurred = _preprocess_image(img)
markers = _find_corner_markers(blurred)
warped = _warp_perspective(img, markers)

cv2.imwrite("warped_omr.jpg", warped)
h, w = warped.shape[:2]

header_height = int(h * 0.18)
grid_top = int(h * 0.22)
grid_height = h - grid_top - int(h * 0.05)
grid_width = int(w * 0.88)
grid_left = int(w * 0.06)

roll_region_w = int(grid_width * 0.35)
roll_region_h = header_height
roll_x = grid_left
roll_y = int(h * 0.05)

set_region_w = int(grid_width * 0.15)
set_region_x = grid_left + int(grid_width * 0.6)

# draw bounding boxes on warped
cv2.rectangle(warped, (roll_x, roll_y), (roll_x + roll_region_w, roll_y + roll_region_h), (0, 0, 255), 10)
cv2.rectangle(warped, (set_region_x, roll_y), (set_region_x + set_region_w, roll_y + roll_region_h), (255, 0, 0), 10)
cv2.rectangle(warped, (grid_left, grid_top), (grid_left + grid_width, grid_top + grid_height), (0, 255, 0), 10)

cv2.imwrite("warped_boxes.jpg", warped)
print("Saved warped_boxes.jpg")
