import cv2
import numpy as np
from app.utils.omr_engine import _preprocess_image, _find_corner_markers, _warp_perspective

def get_dynamic_zones(warped, total_questions=100):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 80000 < area < 2500000:
            boxes.append((x, y, w, h))
            
    # Ensure boxes are unique and filter out fully nested boxes
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    filtered_boxes = []
    for b in boxes:
        x, y, w, h = b
        # We only care about normal table aspect ratios, reject full horizontal lines
        if w > 2000 or h < 200: 
            continue
            
        cx, cy = x + w/2, y + h/2
        is_inside = False
        for fb in filtered_boxes:
            fx, fy, fw, fh = fb
            if fx - 100 < cx < fx+fw + 100 and fy - 100 < cy < fy+fh + 100:
                is_inside = True
                break
        if not is_inside:
            filtered_boxes.append(b)
            
    top_tables = [b for b in filtered_boxes if b[1] < 1400 and b[3] > 600]
    bottom_tables = [b for b in filtered_boxes if b[1] >= 1400 and b[3] > 600]
    
    top_tables.sort(key=lambda b: b[0])
    
    roll_box = None
    set_code_box = None
    
    if len(top_tables) >= 3:
        # X=0: Class, X=378: Roll, X=1095: Subject, X=1480: Set Code
        set_code_box = top_tables[-1] # Right-most tall table
        roll_box = max(top_tables, key=lambda b: b[2]) # Widest tall table
        
    bottom_tables.sort(key=lambda b: b[0])
    
    return {
        "roll": roll_box,
        "set_code": set_code_box,
        "columns": bottom_tables
    }

if __name__ == "__main__":
    img = cv2.imread("omrsheet-board.jpg")
    blurred = _preprocess_image(img)
    markers = _find_corner_markers(blurred)
    if markers is None:
        print("COULD NOT FIND CORNERS")
    else:
        print("Markers found:", markers)
        warped = _warp_perspective(img, markers)
        zones = get_dynamic_zones(warped)
        print("Roll Box:", zones["roll"])
        print("Set Code Box:", zones["set_code"])
        for i, col in enumerate(zones["columns"]):
            print(f"Col {i}: {col}")
