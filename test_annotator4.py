import cv2
import numpy as np

def test_annotator4(img_path):
    print(f"\n--- Testing {img_path} ---")
    img = cv2.imread(img_path)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    # We want to use the exact same logic as process_omr_image
    from app.utils.omr_engine import process_omr_image
    from app.utils.omr_engine import _detect_marked_option_by_density

    # Instead of full process, just run fallback drawing
    from app.utils.omr_engine import _find_markers
    warped = None
    markers = _find_markers(gray)
    if markers is not None and len(markers) == 4:
        print("DEBUG: Corner markers found, applying perspective transform")
        from app.utils.omr_engine import _order_points, _four_point_transform
        rect = _order_points(np.array(markers))
        warped = _four_point_transform(gray, rect)
        
        # Original template aspect ratio (A4 approx)
        if warped.shape[1] > 0:
            aspect_ratio = warped.shape[0] / float(warped.shape[1])
            target_width = 2480
            target_height = int(target_width * aspect_ratio)
            
            # Constrain height to standard A4 ratio if it's too tall/short
            # A4 aspect ratio is ~1.414 (e.g., 3508x2480)
            target_height = min(target_height, 3500)
            warped = cv2.resize(warped, (target_width, target_height))
    else:
        target_width = 2480
        aspect = gray.shape[0] / float(gray.shape[1])
        target_h = int(target_width * aspect)
        warped = cv2.resize(gray, (target_width, target_h))

    thresh = cv2.adaptiveThreshold(
        warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

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

    debug_img = warped.copy()
    if len(debug_img.shape) == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

    total_questions = 30
    rows_count = len(row_centers)
    num_cols = (total_questions + rows_count - 1) // rows_count
    
    for q in range(total_questions):
        cols_count = len(x_grid) // 4
        if cols_count == 0:
            break
            
        c_idx = q // rows_count
        r_idx = q % rows_count
        
        if r_idx >= len(row_centers): continue
            
        cy = row_centers[r_idx]
        
        for opt in range(4):
            x_idx = c_idx * 4 + opt
            if x_idx >= len(x_grid): continue
            cx = x_grid[x_idx]
            
            box_radius_x, box_radius_y = 25, 20
            roi_y1, roi_y2 = max(0, cy - box_radius_y), min(warped.shape[0], cy + box_radius_y)
            roi_x1, roi_x2 = max(0, cx - box_radius_x), min(warped.shape[1], cx + box_radius_x)
            
            cv2.rectangle(debug_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
            cv2.putText(debug_img, f"Q{q+1}", (roi_x1, roi_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite(f"{img_path}_annotated.jpg", debug_img)

test_annotator4('omrsheet.png')
