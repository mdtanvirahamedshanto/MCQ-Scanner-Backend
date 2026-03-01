import cv2
import numpy as np
from app.utils.omr_engine import process_omr_image, _read_roll_number, _read_set_code

def test_inference(img_path):
    print(f"\n--- Analyzing {img_path} ---")
    
    # We will simulate what process_omr_image does, but with our proportional inference inserted
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

    # 1. Run the same bubble logic from fallback to find x_grid and row_centers
    b_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bubble_xs = []
    bubble_ys = []
    for cnt in b_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 800 < area < 12000 and 0.5 < (w / (h + 1e-6)) < 2.0:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circ = 4 * np.pi * (cv2.contourArea(cnt) / (peri * peri))
                if 0.4 < circ <= 1.5:
                    bubble_xs.append(x + w // 2)
                    bubble_ys.append(y + h // 2)
                    
    bubble_ys.sort()
    y_clusters = []
    if bubble_ys:
        current_cluster = [bubble_ys[0]]
        for i in range(1, len(bubble_ys)):
            if bubble_ys[i] - bubble_ys[i-1] < 25:
                current_cluster.append(bubble_ys[i])
            else:
                y_clusters.append(int(np.median(current_cluster)))
                current_cluster = [bubble_ys[i]]
        y_clusters.append(int(np.median(current_cluster)))
        
    b_ys_arr = np.array(bubble_ys)
    valid_y_clusters = []
    for cluster_y in y_clusters:
        count = np.sum(np.abs(b_ys_arr - cluster_y) < 25)
        if count >= 3:
            valid_y_clusters.append(cluster_y)
            
    row_centers = valid_y_clusters
    
    bubble_xs.sort()
    x_clusters = []
    if bubble_xs:
        current_cluster = [bubble_xs[0]]
        for i in range(1, len(bubble_xs)):
            if bubble_xs[i] - bubble_xs[i-1] < 25: 
                current_cluster.append(bubble_xs[i])
            else:
                x_clusters.append(int(np.median(current_cluster)))
                current_cluster = [bubble_xs[i]]
        x_clusters.append(int(np.median(current_cluster)))

    b_xs_arr = np.array(bubble_xs)
    valid_x_clusters = []
    for cluster_x in x_clusters:
        count = np.sum(np.abs(b_xs_arr - cluster_x) < 25)
        if count >= max(1, len(row_centers) * 0.15):
            valid_x_clusters.append(cluster_x)
            
    x_grid = valid_x_clusters

    if not row_centers or not x_grid:
        print("Could not infer MCQ grid.")
        return

    # MCQ grid bounds
    mcq_x_start = x_grid[0]
    mcq_x_end = x_grid[-1]
    mcq_y_start = row_centers[0]
    
    mcq_width = mcq_x_end - mcq_x_start
    # distance between rows
    row_gap = (row_centers[-1] - row_centers[0]) / max(1, len(row_centers) - 1)
    # distance between cols
    col_gap = (x_grid[-1] - x_grid[0]) / max(1, len(x_grid) - 1)
    
    print(f"MCQ Grid: X=({mcq_x_start} to {mcq_x_end}), Y_start={mcq_y_start}")
    print(f"Col gap: {col_gap:.1f}, Row gap: {row_gap:.1f}")

    # Standard offset for top boxes relative to MCQ grid:
    # Set Code is to the right 
    # Roll is to the left
    
    # Actually, we can just trace the columns upwards!
    # Roll Number has 6 digits. It spans 6 * col_gap width.
    # Set Code has 4 options. It spans 4 * col_gap width.
    # Usually Set Code is far right.
    # What if we just dynamically calculate the bounding box based on the known col_gap and row_gap?
    
    # Roll Box estimation
    roll_w = int(col_gap * 6 * 1.5) # slightly larger to be safe
    roll_h = int(row_gap * 10 * 1.2)
    # It usually starts at mcq_x_start, and ends some distance above mcq_y_start
    roll_y_end = mcq_y_start - int(row_gap * 2) # 2 rows of padding
    roll_y_start = roll_y_end - roll_h
    roll_x_start = mcq_x_start
    
    print(f"Inferred Roll Box: x={roll_x_start}, y={roll_y_start}, w={roll_w}, h={roll_h}")
    
    # Set Code estimation
    set_w = int(col_gap * 4 * 1.5)
    set_h = int(row_gap * 1 * 1.5) # 1 row usually? Or 4 rows? Set Code is usually A B C D in a row.
    # Wait, some Set Codes are 1 column 4 rows.
    # Assuming Bengali codes: ক, খ, গ, ঘ -> 4 options. Usually horizontal or vertical?
    
    roll = _read_roll_number(warped, roll_x_start, roll_y_start, roll_w, roll_h)
    print(f"Read Roll Number: {roll}")
    
for img in ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']:
    test_inference(img)
