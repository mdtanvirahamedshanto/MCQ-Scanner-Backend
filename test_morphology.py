import cv2
import numpy as np

def test_morphology_zones(img_path):
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
    
    # --- MORPHOLOGICAL FIX ---
    # Create horizontal kernel and vertical kernel
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine lines to form strict table grids
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    
    # Connect slightly broken intersections
    kernel = np.ones((5,5), np.uint8)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 80000 < w * h < 2500000:
            boxes.append((x, y, w, h))
            
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    
    filtered_boxes = []
    for b in boxes:
        x, y, w, h = b
        if w > 2000 or h < 200: 
            continue
            
        cx, cy = x + w/2, y + h/2
        is_inside = False
        for fb in filtered_boxes:
            fx, fy, fw, fh = fb
            if fx - 50 < cx < fx+fw + 50 and fy - 50 < cy < fy+fh + 50:
                is_inside = True
                break
        if not is_inside:
            filtered_boxes.append(b)

    print(f"Found {len(filtered_boxes)} filtered boxes:")
    for b in filtered_boxes:
        print(f"  {b} (Area: {b[2]*b[3]})")

for img in ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']:
    test_morphology_zones(img)
