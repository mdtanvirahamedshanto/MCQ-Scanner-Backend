import cv2
import numpy as np

def analyze_top_bubbles(img_path):
    print(f"\n--- Analyzing {img_path} ---")
    img = cv2.imread(img_path)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    target_width = 2480
    aspect = gray.shape[0] / float(gray.shape[1])
    target_h = int(target_width * aspect)
    warped = cv2.resize(gray, (target_width, target_h))

    thresh = cv2.adaptiveThreshold(
        warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    top_part = thresh[0:int(target_h * 0.4), :]
    b_contours, _ = cv2.findContours(top_part, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bubble_xs = []
    bubble_ys = []

    for cnt in b_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Even more relaxed constraints to catch empty bubbles
        if 500 < area < 15000 and 0.4 < (w / (h + 1e-6)) < 2.5:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circ = 4 * np.pi * (cv2.contourArea(cnt) / (peri * peri))
                if 0.3 < circ <= 2.0:
                    bubble_xs.append(x + w // 2)
                    bubble_ys.append(y + h // 2)

    print(f"DEBUG: Found {len(bubble_xs)} bubbles in the top section.")

    # Cluster X coordinates
    bubble_xs.sort()
    x_clusters = []
    if bubble_xs:
        current_cluster = [bubble_xs[0]]
        for i in range(1, len(bubble_xs)):
            if bubble_xs[i] - bubble_xs[i-1] < 30: 
                current_cluster.append(bubble_xs[i])
            else:
                x_clusters.append(int(np.median(current_cluster)))
                current_cluster = [bubble_xs[i]]
        x_clusters.append(int(np.median(current_cluster)))

    print(f"X clusters ({len(x_clusters)}): {x_clusters}")

    # Cluster Y coordinates
    bubble_ys.sort()
    y_clusters = []
    if bubble_ys:
        current_cluster = [bubble_ys[0]]
        for i in range(1, len(bubble_ys)):
            if bubble_ys[i] - bubble_ys[i-1] < 30: 
                current_cluster.append(bubble_ys[i])
            else:
                y_clusters.append(int(np.median(current_cluster)))
                current_cluster = [bubble_ys[i]]
        y_clusters.append(int(np.median(current_cluster)))

    print(f"Y clusters ({len(y_clusters)}): {y_clusters}")

for img in ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']:
    analyze_top_bubbles(img)
