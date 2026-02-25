import cv2
import numpy as np
from app.utils.omr_engine import _preprocess_image, _find_corner_markers

img = cv2.imread("omrsheet.png")
blurred = _preprocess_image(img)
markers = _find_corner_markers(blurred)

if markers is None:
    aspect = gray.shape[0] / float(gray.shape[1])
    target_h = int(2480 * aspect)
    warped = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (2480, target_h))
    warped_color = cv2.resize(img, (2480, target_h))
else:
    from app.utils.omr_engine import _warp_perspective
    warped = _warp_perspective(img, markers)
    warped_color = warped.copy()
    if len(warped.shape) == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(
    warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)

# Extract timing marks to get row clusters
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
t_marks = []
h_img, w_img = warped.shape[:2]
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 800 < w * h < 12000 and x < w_img * 0.15:
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(cv2.contourArea(cnt)) / hull_area
            if solidity > 0.9 and 1.0 < (w / h) < 2.0:
                t_marks.append((x, y, w, h))

t_marks.sort(key=lambda m: m[1])
row_centers = [y + h // 2 for (x, y, w, h) in t_marks]

# Extract bubbles
b_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
bubble_xs = []
for cnt in b_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if 3000 < area < 8000 and 0.8 < (w / (h + 1e-6)) < 1.2:
        peri = cv2.arcLength(cnt, True)
        if peri > 0:
            circ = 4 * np.pi * (cv2.contourArea(cnt) / (peri * peri))
            if 0.8 < circ <= 1.2:
                bubble_xs.append(x + w // 2)

bubble_xs.sort()
clusters = []
current_cluster = [bubble_xs[0]]
for i in range(1, len(bubble_xs)):
    if bubble_xs[i] - bubble_xs[i-1] < 20: 
        current_cluster.append(bubble_xs[i])
    else:
        clusters.append(int(np.median(current_cluster)))
        current_cluster = [bubble_xs[i]]
clusters.append(int(np.median(current_cluster)))

b_xs_arr = np.array(bubble_xs)
valid_clusters = []
for cluster_x in clusters:
    count = np.sum(np.abs(b_xs_arr - cluster_x) < 20)
    if count >= max(1, len(row_centers) * 0.25):
        valid_clusters.append(cluster_x)
        
valid_clusters.sort()
x_grid = valid_clusters[:len(row_centers) * 4]

# Draw simulated marks
expected_answers = []
for q in range(30): # Hardcoded for this sheet
    c_idx = q // len(row_centers)
    r_idx = q % len(row_centers)
    cy = row_centers[r_idx]
    
    ans = q % 4  # 0, 1, 2, 3 pattern
    cx = x_grid[c_idx * 4 + ans]
    cv2.circle(warped_color, (cx, cy), 20, (0, 0, 0), -1)
    expected_answers.append(ans)

cv2.imwrite("simulated_marked.jpg", warped_color)
print("Saved simulated_marked.jpg")
print("Expected Answers:", expected_answers)
