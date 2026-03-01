import sys
import cv2
import numpy as np
from app.utils.omr_engine import process_omr_image, OMRProcessor

def test_new_sheet():
    img_path = 'omrsheet.png'
    print(f"\n--- Testing New Sheet Template: {img_path} ---")
    try:
        # First, let's see what happens with default 30 questions
        result = process_omr_image(img_path, num_questions=30)
        print(f"Success: {result.success}")
        if not result.success:
            print(f"Error: {result.error_message}")
            
        print(f"Roll Number: {result.roll_number}")
        print(f"Set Code: {result.set_code}")
        
        print(f"Extracted Answers List (length {len(result.answers)}):")
        # Group them into columns of 10
        for i in range(10):
            col1 = f"Q{i+1}: {result.answers[i]}" if i < len(result.answers) else ""
            col2 = f"Q{i+11}: {result.answers[i+10]}" if i+10 < len(result.answers) else ""
            col3 = f"Q{i+21}: {result.answers[i+20]}" if i+20 < len(result.answers) else ""
            print(f"  {col1:<15} {col2:<15} {col3:<15}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

test_new_sheet()
