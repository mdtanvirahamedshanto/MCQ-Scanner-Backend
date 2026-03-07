import cv2
import numpy as np
from app.utils.omr_engine import _find_corner_markers, _warp_perspective

img = cv2.imread("mcq.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
markers = _find_corner_markers(gray)
warped = _warp_perspective(img, markers)
gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray_w, 180, 255, cv2.THRESH_BINARY_INV)

# Row 11 (y=925)
cy = 925
# Left column peaks
roi_l = binary[cy-20:cy+20, 200:1200]
sums_l = np.sum(roi_l, axis=0)
peaks_l = []
for i in range(1, len(sums_l)-1):
    if sums_l[i] > sums_l[i-1] and sums_l[i] > sums_l[i+1] and sums_l[i] > 1000:
        peaks_l.append(200 + i)
filtered_l = []
for p in peaks_l:
    if not filtered_l or p - filtered_l[-1] > 50: filtered_l.append(p)

# Right column peaks
roi_r = binary[cy-20:cy+20, 1500:2480]
sums_r = np.sum(roi_r, axis=0)
peaks_r = []
for i in range(1, len(sums_r)-1):
    if sums_r[i] > sums_r[i-1] and sums_r[i] > sums_r[i+1] and sums_r[i] > 1000:
        peaks_r.append(1500 + i)
filtered_r = []
for p in peaks_r:
    if not filtered_r or p - filtered_r[-1] > 50: filtered_r.append(p)

print("Left peaks:", filtered_l)
print("Right peaks:", filtered_r)
