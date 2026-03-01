import cv2
import numpy as np

def debug_timing_marks(img_path):
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
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    t_marks = []
    
    h_img, w_img = warped.shape[:2]
    
    print("Potential timing mark contours (x < 15%, 500 < area < 20000):")
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < w_img * 0.15:
            area = w * h
            if 500 < area < 20000:
                aspect_ratio = w / float(h + 1e-6)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(cv2.contourArea(cnt)) / hull_area
                    # Let's print the candidates
                    print(f"  Box: x={x}, y={y}, w={w}, h={h}, area={area}, aspect={aspect_ratio:.2f}, solidity={solidity:.2f}")

for img in ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']:
    debug_timing_marks(img)
