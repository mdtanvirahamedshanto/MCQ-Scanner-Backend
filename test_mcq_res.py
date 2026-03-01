import cv2

img = cv2.imread('mcq.jpeg')
print(f"mcq.jpeg shape: {img.shape}")
print(f"mcq.jpeg area: {img.shape[0] * img.shape[1]}")

from app.utils.omr_engine import *
gray = _preprocess_image(img)
h, w = gray.shape
print(f"gray shape: {gray.shape}, area: {h*w}")
