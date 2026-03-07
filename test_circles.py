import cv2
import numpy as np
from app.utils.omr_engine import _find_corner_markers, _warp_perspective

def find_circles():
    img = cv2.imread("mcq.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    markers = _find_corner_markers(gray)
    warped = _warp_perspective(img, markers)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Let's crop to row 11 (cy=925) and just the right half
    roi = gray[925-40:925+40, 1500:2400]
    
    # Apply HoughCircles
    circles = cv2.HoughCircles(
        roi, cv2.HOUGH_GRADIENT, dp=1, minDist=80,
        param1=50, param2=20, minRadius=15, maxRadius=35
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        xs = sorted([c[0] + 1500 for c in circles[0,:]])
        print("Found bubble X coordinates on row 11 right half:")
        print(xs)
        
        # Draw for debugging
        roi_color = warped[925-40:925+40, 1500:2400].copy()
        for i in circles[0,:]:
            cv2.circle(roi_color,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.imwrite("row11_circles.jpg", roi_color)
    else:
        print("No circles found.")

if __name__ == "__main__":
    find_circles()
