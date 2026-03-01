import cv2
import numpy as np

def _get_bubble_density(roi: np.ndarray) -> float:
    if roi.size == 0:
        return 0.0
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return np.sum(binary > 0) / binary.size

def find_marked_top_bubbles(img_path):
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
    
    # In practice, we know MCQ grid starts around Y=1500 for mcq.jpeg, Y=1400 etc.
    # Let's just look at the top 40% of the image.
    top_limit = int(target_h * 0.45)
    
    b_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    marked_bubbles = []

    for cnt in b_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y > top_limit:
            continue
            
        area = w * h
        # We are looking for FILLED bubbles, which are blobby
        if 800 < area < 8000 and 0.5 < (w / (h + 1e-6)) < 2.0:
            roi = warped[max(0, y):min(target_h, y+h), max(0, x):min(target_width, x+w)]
            density = _get_bubble_density(roi)
            if density > 0.4: # Only highly dense/filled bubbles
                marked_bubbles.append((x + w//2, y + h//2, density, w, h))

    print(f"Found {len(marked_bubbles)} marked bubbles in top section:")
    
    # Simple deduplication (contours can trigger multiple times inside/outside)
    unique_bubbles = []
    for cx, cy, d, w, h in sorted(marked_bubbles, key=lambda b: b[2], reverse=True):
        duplicate = False
        for ux, uy, ud, uw, uh in unique_bubbles:
            if abs(cx - ux) < 30 and abs(cy - uy) < 30:
                duplicate = True
                break
        if not duplicate:
            unique_bubbles.append((cx, cy, d, w, h))
            
    # Sort left to right
    unique_bubbles.sort(key=lambda b: b[0])
    
    for b in unique_bubbles:
        print(f"  X={b[0]}, Y={b[1]}, Density={b[2]:.2f}")

for img in ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']:
    find_marked_top_bubbles(img)
