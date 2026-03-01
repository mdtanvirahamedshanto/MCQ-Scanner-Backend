import sys
from app.utils.omr_engine import process_omr_image
import logging
logging.basicConfig(level=logging.INFO)

images = ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']

for img in images:
    print(f"Testing {img}...")
    try:
        result = process_omr_image(img, num_questions=30)
        print(f"Success: {result.success}")
        print(f"Error: {result.error_message}")
        print(f"Roll: {result.roll_number}")
        print(f"Set Code: {result.set_code}")
    except Exception as e:
        print(f"Exception: {e}")
    print("-" * 20)
