import cv2
import numpy as np

# A duplicate of warp function from omr_engine
def _warp_perspective(img, markers):
    tl, tr, bl, br = markers
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # force exact dimensions for consistency if we know it
    # But let's just use the computed max width/height
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(markers, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

from app.utils.omr_engine import process_omr_image
# We can just hijack process_omr_image by importing `_warp_perspective` and `_find_document_corners` locally
from app.utils.omr_engine import _find_document_corners

img = cv2.imread("mcq.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)
markers = _find_document_corners(edged)

warped = _warp_perspective(img, markers)
gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray_w, 180, 255, cv2.THRESH_BINARY_INV)

print(f"Warped shape: {binary.shape}")
cy = 925
roi = binary[cy-20:cy+20, 1500:2286]
vert_sum = np.sum(roi, axis=0)

smoothed = np.convolve(vert_sum, np.ones(5)/5, mode='same')
peaks = []
for i in range(1, len(smoothed)-1):
    if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > 1000: # at least half pixels black
        peaks.append(1500 + i)

filtered = []
for p in peaks:
    if not filtered or p - filtered[-1] > 30:
        filtered.append(p)
        
print("Right column peaks:", filtered)
