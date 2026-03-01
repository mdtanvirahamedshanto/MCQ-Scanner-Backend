import sys
import cv2
from app.utils.omr_engine import process_omr_image
import traceback

images = ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']

for img in images:
    print(f"Testing {img}...")
    try:
        result = process_omr_image(img, num_questions=30)
        print(f"Success: {result.success}")
        print(f"Answers: {result.answers}")
        
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
    print("-" * 20)
