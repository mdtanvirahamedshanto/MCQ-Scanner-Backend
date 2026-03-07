import cv2
import numpy as np
from app.utils.omr_engine import _find_corner_markers, _warp_perspective

img = cv2.imread("mcq.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
markers = _find_corner_markers(gray)
warped = _warp_perspective(img, markers)
gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# Row 1 is around y=925. Let's take a slice y=[900, 950]
slice = gray_w[900-30:925+30, :]
# Threshold it
_, binary = cv2.threshold(slice, 180, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
centers = []
for c in contours:
    M = cv2.moments(c)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.contourArea(c)
        if 800 < area < 3000: # Bubble area
            centers.append(cx)

centers.sort()
print("All detected bubble centers in Row 1 (unfiltered):")
print(centers)

# We expect 4 in the left half and 4 in the right half
left_half = [c for c in centers if c < 1250]
right_half = [c for c in centers if c > 1250]

print("Left centers:", left_half)
print("Right centers:", right_half)
