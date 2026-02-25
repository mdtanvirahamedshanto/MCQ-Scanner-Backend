import cv2
import numpy as np

img = cv2.imread("warped_omr.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if 100000 < area < 2000000:  # Big enough to be a table, small enough to not be the page
        # To avoid nested duplicates (since lines have thickness), 
        # check if this box is very similar to an existing one
        is_duplicate = False
        for bx, by, bw, bh in boxes:
            if abs(x-bx) < 20 and abs(y-by) < 20 and abs(w-bw) < 20 and abs(h-bh) < 20:
                is_duplicate = True
                break
        if not is_duplicate:
            boxes.append((x, y, w, h))

print(f"Found {len(boxes)} unique large tables.")

# Sort tables by Y entirely to separate top half (Info) from bottom half (Questions)
boxes.sort(key=lambda b: b[1])

for i, (x, y, w, h) in enumerate(boxes):
    print(f"Table {i}: X={x}, Y={y}, W={w}, H={h} (Area={w*h})")
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
    cv2.putText(img, str(i), (x+10, y+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

cv2.imwrite("tables_v3.jpg", img)
print("Saved tables_v3.jpg")
