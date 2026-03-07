import cv2
import numpy as np

img = cv2.imread("mcq.png")
print(f"Original image shape: {img.shape}")

# Find black squares for corner markers
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

markers = []
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        if 40 < w < 200 and 40 < h < 200:
            aspect = float(w) / h
            if 0.8 <= aspect <= 1.2:
                markers.append((x+w//2, y+h//2))
print("Markers found at:", markers)
