import cv2
import numpy as np
from app.utils.omr_engine import extract_omr_roi

def find_x():
    img = cv2.imread("mcq.png")
    warped_color = extract_omr_roi(img)
    if warped_color is None:
        print("Could not warp")
        return
        
    gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Sum pixel intensities vertically in the row for Q11 (y=925)
    cy = 925
    roi = binary[cy-20:cy+20, 1600:2286]
    vert_sum = np.sum(roi, axis=0)
    
    # Smooth the sums
    smoothed = np.convolve(vert_sum, np.ones(10)/10, mode='same')
    
    peaks = []
    for i in range(1, len(smoothed)-1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > 2000:
            peaks.append(1600 + i)
            
    print("Found peaks in right column (rough X coords):")
    
    # Filter out close peaks
    filtered = []
    for p in peaks:
        if not filtered or p - filtered[-1] > 50:
            filtered.append(p)
            
    print(filtered)

if __name__ == "__main__":
    find_x()
