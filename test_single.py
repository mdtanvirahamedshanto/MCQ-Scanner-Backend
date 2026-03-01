import sys
import os
import cv2

# Ensure backend directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.omr_engine import OMRProcessor

def test_image(img_path):
    processor = OMRProcessor(total_questions=30)
    print(f"\n{'='*40}")
    print(f"Testing Extract & Evaluate for: {img_path}")
    print(f"{'='*40}")
    
    try:
        result = processor.process(img_path)
        print(f"Success: {result.success}")
        if result.error_message:
            print(f"Error: {result.error_message}")
            
        print(f"Roll Number: {result.roll_number}")
        print(f"Set Code: {result.set_code}")
        
        print("\nExtracted Answers:")
        for i in range(30):
            ans = result.answers[i] if i < len(result.answers) else -1
            print(f"  Q{i+1}: {ans}")
            
    except Exception as e:
        print(f"Processing crashed: {str(e)}")

if __name__ == '__main__':
    test_image('mcq.jpeg')
