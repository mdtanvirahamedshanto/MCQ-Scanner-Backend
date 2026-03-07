import cv2
import numpy as np
from app.utils.omr_engine import _find_corner_markers, _warp_perspective

def measure():
    img = cv2.imread("mcq.png")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    markers = _find_corner_markers(gray)

    warped = _warp_perspective(img, markers)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    cy = 925 # row 11
    roi = binary[cy-20:cy+20, 1500:2480]
    vert_sum = np.sum(roi, axis=0)

    smoothed = np.convolve(vert_sum, np.ones(5)/5, mode='same')
    peaks = []
    for i in range(1, len(smoothed)-1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > 1000:
            peaks.append(1500 + i)

    filtered = []
    for p in peaks:
        if not filtered or p - filtered[-1] > 30:
            filtered.append(p)
            
    print("Found right column X coordinates in warped image:")
    print(filtered)

if __name__ == "__main__":
    measure()
