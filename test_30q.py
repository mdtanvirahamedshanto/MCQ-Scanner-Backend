import sys
from app.utils.omr_engine import process_omr_image

images = ['mcq.jpeg', 'mcq2.jpeg', 'mcq3.jpeg']

for img in images:
    print(f"\n--- Testing 30 Questions on {img} ---")
    try:
        result = process_omr_image(img, num_questions=30)
        print(f"Success: {result.success}")
        if not result.success:
            print(f"Error: {result.error_message}")
        print(f"Extracted Answers List (length {len(result.answers)}):")
        
        # Group them into columns of 10 for easier reading
        for i in range(10):
            col1 = f"Q{i+1}: {result.answers[i]}" if i < len(result.answers) else ""
            col2 = f"Q{i+11}: {result.answers[i+10]}" if i+10 < len(result.answers) else ""
            col3 = f"Q{i+21}: {result.answers[i+20]}" if i+20 < len(result.answers) else ""
            print(f"  {col1:<15} {col2:<15} {col3:<15}")
    except Exception as e:
        print(f"Exception: {e}")
