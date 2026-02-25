import cv2
import numpy as np

img = cv2.imread("warped_omr.jpg", cv2.IMREAD_GRAYSCALE)
thresh = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

sizes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if area > 50000:
        sizes.append((area, x, y, w, h))

sizes.sort(reverse=True)
print("Largest contours:")
for s in sizes[:10]:
    print(f"Area={s[0]:.0f}, Pos=({s[1]}, {s[2]}), Size={s[3]}x{s[4]}")

