import cv2
from app.utils.omr_engine import _preprocess_image, _check_image_blur

images = ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']
for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_path}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"{img_path} Laplacian Var: {laplacian_var}")
