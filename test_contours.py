import cv2
import numpy as np

img = cv2.imread('mcq2.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
target_width = 2480
aspect = gray.shape[0] / float(gray.shape[1])
target_h = int(target_width * aspect)
warped = cv2.resize(gray, (target_width, target_h))

thresh = cv2.adaptiveThreshold(
    warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    boxes.append((w * h, x, y, w, h))

boxes.sort(key=lambda x: x[0], reverse=True)
print("Top 20 largest contours by area:")
for i, b in enumerate(boxes[:20]):
    print(f"  {i}: Area={b[0]}, Box=(x={b[1]}, y={b[2]}, w={b[3]}, h={b[4]})")
