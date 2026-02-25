import cv2
import numpy as np

img = cv2.imread("warped_omr.jpg", cv2.IMREAD_GRAYSCALE)
thresh = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)

# Find vertical lines
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=1)

# Find horizontal lines
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)

# Combine
table_mask = cv2.addWeighted(v_lines, 0.5, h_lines, 0.5, 0.)
_, table_mask = cv2.threshold(table_mask, 128, 255, cv2.THRESH_BINARY)

# Find contours of tables/grids
contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

res = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area > 10000:
        res.append((area, x, y, w, h))

res.sort(key=lambda item: item[1]) # Sort by X to go left to right
for i, item in enumerate(res):
    print(f"Grid {i}: Area={item[0]:.0f}, Pos=({item[1]}, {item[2]}), Size={item[3]}x{item[4]}")
