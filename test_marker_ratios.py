import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.utils.omr_engine import *

from app.utils.omr_engine import _preprocess_image, _find_corner_markers

def check_ratio(img_path):
    img = cv2.imread(img_path)
    if img is None: return
    blur = _preprocess_image(img)
    markers = _find_corner_markers(blur)
    h, w = blur.shape
    total_area = h * w
    print(f"\n{img_path}: Area={total_area}")
    if markers is not None:
        x, y = markers[:, 0], markers[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        print(f"Quad Area={area}, Ratio={area/total_area:.2f}")
    else:
        print("No markers.")

check_ratio('mcq.jpeg')
check_ratio('omrsheet.png')
