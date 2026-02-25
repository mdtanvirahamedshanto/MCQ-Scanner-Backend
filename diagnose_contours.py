import cv2
import numpy as np

img = cv2.imread("test_omr.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)
edges = cv2.Canny(gray, 50, 150)
combined = cv2.bitwise_or(thresh, edges)
kernel = np.ones((3, 3), np.uint8)
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(
    combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

h, w = gray.shape
print(f"Image shape: {w}x{h}")
total_area = h * w

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100: # at least some small size
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            x, y, w_rect, h_rect = cv2.boundingRect(approx)
            aspect = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1e-6)
            if 0.5 < aspect < 2.0:
                print(f"Candidate - Area: {area:.1f}, Pos: ({x}, {y}), Size: {w_rect}x{h_rect}")
